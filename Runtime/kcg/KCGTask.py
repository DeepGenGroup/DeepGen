# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

import logging
from typing import List,Type,Tuple
from kcg.Utils import *
from kcg.Kernel import *
from kcg.CompiledKernelFactory import *
from kcg.Operators import matmul
import sys
import numpy as np
from kcg.KCGCompiler import KCGCompiler
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from logging import *
from typing import List, Tuple
import glob
import ctypes
import json
from ConfigGenerator import ParseTuningSpace
from RemoteUtils import *
import socket

# import pymlir

def g_getBaselineStoreFileName(devId) -> str:
    return PathManager().default_cache_dir() + f'/BenchmarkTorchEps_{devId}.log'

def g_getBaselinePklPath(devid) -> str :
    return PathManager().default_cache_dir() + f'/gemmBase_{devid}.pkl'

class KernelTestResult :
    def __init__(self,kpm : KernelArgMatmul):
        self.kpm = kpm
        self.isCorrect = False
        self.acc = 0.0
        self.kcg_elapseTimeMs = 0.0
        self.torch_elapseTimeMs = 0.0
        self.diffRate = 0.0
        self.maxError = 0.0
    def __str__(self):
        return "{" + f"correct : {self.isCorrect}, acc : {self.acc}, torchMs : {self.torch_elapseTimeMs}], config : {self.kpm}" + "}"
    def jsonfy(self) -> Dict :
        obj = { "correct" : self.isCorrect, 
                "acc" : self.acc,
                "torchMs" : self.torch_elapseTimeMs,
                "kcgMs" : self.kcg_elapseTimeMs,
                "config" : self.kpm.jsonfy()
            }
        return obj
    
    def parseFromJson(self,jsonObj) :
        self.isCorrect = jsonObj['correct']
        self.acc = jsonObj['acc']
        self.torch_elapseTimeMs = jsonObj['torchMs']
        self.kcg_elapseTimeMs = jsonObj['kcgMs']
        self.kpm = KernelArgMatmul(0,0,0,0,1,1,1)
        self.kpm.assignWithJson(jsonObj['config'])


class PerfTester :
    def __init__(self,devId:int,atol:float,rtol:float,nTorchEpsInitTest = 50, baselineInitializer : OperatorBaseArgs = None):
        self.matA = None
        self.matB = None
        self.matC = None
        self.matD = None
        self.torch_eps = -1.0  # torch的eps，用于计算 speedup
        self.BestPerf = [] #: List[KernelTestResult]
        self.torchDynamicEps = []  # torch的动态eps，用于描述torch的性能变化（卡的稳定性）
        self.check_dynamic_torch_perf = 2000  # 每执行完多少个case，检查一下torch的当前性能。记录波动
        self._torchEpsStoreFile = g_getBaselineStoreFileName(devId)
        self._devId = devId
        self._atol = atol
        self._rtol = rtol
        self._isInited = False
        self.nTorchEpsInitTestCount = nTorchEpsInitTest
        self.controllerProc = None
        self.workFlag = ParallelTaskManager.ctx.Manager().Value(ctypes.c_int,1)  # continue glob pkl
        self.initArgJsonName = f"init_arg_{self._devId}.json"

        self.Process = ParallelTaskManager.Process
        self.baselineInitializer = baselineInitializer
    
    @staticmethod
    def _controllerRemote(workflag,finishflag,perflogPath : str) :
        server = MyTCPServer()
        server.listen()
        msg = ""
        initArgRcvMark = 'INIT='
        while finishflag.value <= 0 :
            if workflag.value > 0 :
                msg = server.recv()
            if msg.find('EXIT') != -1 :
                workflag.value = 0
                print("== recv EXIT message, waiting for last batch pkls testing ok ... ")
            time.sleep(0.5)
        print("== controller reply perflogpath and ready to stop ... ")
        server.reply(perflogPath)
        
            
    def _startController(self,perflogpath,finishflag) :
        if self.controllerProc is None :
            self.controllerProc = self.Process(target=PerfTester._controllerRemote,args=(self.workFlag,finishflag, perflogpath))
            self.controllerProc.start()
    
    def init_cuda(self) :
        DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
        if not self._isInited :
            print("init_cuda devid=",self._devId)
            DeviceInfo.set_visible_devices([self._devId])
            DeviceInfo.set_current_device(self._devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
            if not torch.cuda.is_available() :
                torch.cuda.init()
                torch.cuda.empty_cache()
            self._isInited = True
    
    def _compare_with_error(self, tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
        abs_diff = torch.abs(tensor1 - tensor2)
        rel_diff = abs_diff / (torch.abs(tensor1) + 1e-5)  # 避免除以零的情况
        # 比较绝对误差和相对误差
        error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
        diff_elements = torch.sum(error_mask).item()
        max_error = torch.max(torch.abs(tensor1 - tensor2))
        return diff_elements, max_error
    
    def _init_AB(self, kpm:KernelArgMatmul, inConfig:UserInputs) :
        if kpm.batch > 1:
            self.matA = torch.randn(kpm.batch,kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device=f'cuda:{self._devId}')
            self.matB = torch.randn(kpm.batch,kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device=f'cuda:{self._devId}')
        else:
            self.matA = torch.randn(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device=f'cuda:{self._devId}')
            self.matB = torch.randn(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device=f'cuda:{self._devId}')
    
    def _init_AB_with_detailed_info(self, batch, m,n,k, datatype : torch.dtype) :
        if batch > 1:
            self.matA = torch.randn(batch,m,k, dtype= datatype, device=f'cuda:{self._devId}')
            self.matB = torch.randn(batch,k,n, dtype= datatype, device=f'cuda:{self._devId}')
        else:
            self.matA = torch.randn(m,k, dtype= datatype, device=f'cuda:{self._devId}')
            self.matB = torch.randn(k,n, dtype= datatype, device=f'cuda:{self._devId}')
    
    def inner_test_torch(self,matrixA:torch.Tensor, matrixB:torch.Tensor) -> Tuple[torch.Tensor,float]:
        torchMM = torch.matmul
        if len(matrixA.shape) > 2 :
            torchMM = torch.bmm
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        self.matD = torchMM(matrixA, matrixB)
        ev_end.record()
        torch.cuda.synchronize()
        eps = ev_start.elapsed_time(ev_end)
        return (self.matD, eps)
    
    def _init_torch_eps(self) :
        self.matD, tempEps = self.inner_test_torch(self.matA, self.matB)
        if not self._read_torch_eps_from_file() :
            eps_torch_list = []
            for i in range(0, self.nTorchEpsInitTestCount) :
                self.matD, eps_torch = self.inner_test_torch(self.matA, self.matB)
                eps_torch_list.append(eps_torch)
                self.torch_eps = np.median(eps_torch_list)
        with open(self._torchEpsStoreFile,'w') as f :
            f.write(str(self.torch_eps))
        print(f'======== torch baseline on dev{self._devId} init OK! result written to {self._torchEpsStoreFile} ========= ')
    
    def _read_torch_eps_from_file(self) :
        try :
            with open(self._torchEpsStoreFile,'r+') as f:
                self.torch_eps = float(f.readline()) 
        except Exception as e:
            return False
        except IOError as e:
            return False
        return True
        
    def _inner_test_kcg(self, a : torch.tensor, b : torch.tensor, c : torch.tensor, 
                        packedKernel : CompiledKernel,
                        start_event : torch.cuda.Event, end_event : torch.cuda.Event) :
        start_event.record()
        packedKernel.run(a,b,c)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return c,elapsed_time
    
    def _test_perf(self, kpm:KernelArgMatmul, inConfig : UserInputs, packedKernel : CompiledKernel, 
                   benchmarkCount = 5, warmupCount = 1 ) -> KernelTestResult:
        # self.init_cuda()
        if self.matA is None or self.matB is None :
            self._init_AB(kpm,inConfig)
        result = KernelTestResult(kpm)
        packedKernel.setDevice(0)  # when __init__, env has been set to actual device id. set 0 here
        if kpm.batch > 1:
            self.matC = torch.empty(kpm.batch,kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device=f'cuda:{self._devId}')
        else:
            self.matC = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device=f'cuda:{self._devId}')
        d0,d1 = 0,1
        if len(self.matA.shape) == 3 :
            d0,d1 = 1,2
        atrans = torch.transpose(self.matA,d0,d1).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
        assert(self.matA.is_contiguous())
        assert(self.matB.is_contiguous())
        assert(atrans.is_contiguous())
        if kpm.batch > 1:
            b, M, K = self.matA.shape
            b, K, N = self.matB.shape
        res = []
        aUse = None
        
        if kpm.isATranspose :
            aUse = atrans
        else:
            aUse = self.matA
        # warmup
        torchMM = torch.matmul
        if kpm.batch > 1:
            torchMM = torch.bmm
        for i in range(0,warmupCount) : 
            torchMM(self.matA, self.matB)
            packedKernel.run(aUse, self.matB, self.matC)

        # 计算torch的eps
        if self.torch_eps <= 0 or self.matD is None:
            self._init_torch_eps()
        
        # benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for i in range(0,benchmarkCount) : 
            self.matC,eps = self._inner_test_kcg(aUse, self.matB, self.matC, packedKernel, start_event, end_event)
            res.append(eps)
        print("c=",self.matC)

        if torch.allclose(self.matC, self.matD, atol=self._atol, rtol=self._rtol):
            print('test correct!')
            result.isCorrect = True
            result.torch_elapseTimeMs = self.torch_eps
            result.kcg_elapseTimeMs = np.median(res)
            print(f"median time(ms) : Deepgen = {result.kcg_elapseTimeMs} , PyTorch= {result.torch_elapseTimeMs}")
            result.acc = result.torch_elapseTimeMs/result.kcg_elapseTimeMs
            print(f"speed up: {result.acc}")
        else:
            result.isCorrect = False
            diff,max_error= self._compare_with_error(self.matD, self.matC)
            result.maxError = max_error
            result.diffRate = diff/(M*N)
            print(f'test fail! maxerror={max_error}, diffrate={result.diffRate}')
        packedKernel.deleteBinary()
        return result
    
    def _getBestPerf(self, perfData : List[KernelTestResult], topNum = 1) -> List[KernelTestResult]:
        for data in perfData:
            if data.isCorrect :
                self.BestPerf.append(data)
                self.BestPerf.sort(key=lambda x: x.acc, reverse=True)
                if len(self.BestPerf) > topNum :
                    self.BestPerf = self.BestPerf[0:topNum]
        return self.BestPerf
      
    def jsonfyBestPerfs(self) -> Dict :
        obj = {"results" : []}
        for r in self.BestPerf :
            obj["results"].append(r.jsonfy())
        return obj
    
    def check_torch_dynamic_perf(self,torchPerfLogName,index) :
        t = []
        for i in range(0,10) :
            res,eps = self.inner_test_torch()
            t.append(eps)
        new_torchEps = np.median(t)
        self.torchDynamicEps.append(new_torchEps)
        with open(torchPerfLogName, mode = 'a+') as ff :
            ff.write(f'[{index}] - {new_torchEps};\n')
        return new_torchEps
    
    def runPerfTests(self, pathLock, endsignal,finishflag ,outputPAth = None, benchmarkCount = 5, warmupCount = 1, topNum = 6, torchDynamicLogPath = '', nTorchEpsInitTest = 50, remoteTester : RemoteSSHConnect = None, isAsRemoteTester = False) : 
        # collect kernels from pkl         
        valid_kernels = [] # List[Tuple[KernelArgMatmul,UserInputs,CompiledKernel]]
        total_kernel_count = 0
        dyTorchCounter = 0
        startFlag = True
        if isAsRemoteTester :
            # wait "upload finish or EXIT" signal from remote
            self._startController(outputPAth,finishflag)
        socket_client = None
        if remoteTester is not None :
            # use remote benchmark, connect remoteTester and send initializer args of different tasks
            if remoteTester.connect():
                print(f"connect remotePerfTester success : destip={remoteTester.host}")
                if self.baselineInitializer is not None and self.baselineInitializer.operatorKind != EnumOperator.Invalid :
                    initargJsonPath = PathManager.default_cache_dir() + "/" + self.initArgJsonName
                    self.baselineInitializer.dumpToJson(initargJsonPath)
                    remoteTester.upload_file(initargJsonPath, PathManager.default_cache_dir())
            socket_client = MyTCPClient()
            connected = False
            for i in range(8):
                connected = socket_client.connect(remoteTester.host)
                if connected :
                    break
                time.sleep(5)
            if not connected :
                assert False, f"[Fatal] connect tcpserver failed : destip={remoteTester.host}"
            else:
                print(f"[I] connect tcpserver success! destip={remoteTester.host}")
        else:
            # run local benchmark
            self.init_cuda()
            # wait init arg file upload to dir
            while startFlag :
                argfile = glob.glob(PathManager.default_cache_dir() +"/"+ self.initArgJsonName)
                if len(argfile) <= 0:
                    time.sleep(1)
                else:
                    break
            self.baselineInitializer.parseFromJsonfile(argfile[0])
            arglist = self.baselineInitializer.argList
            b = arglist[0]
            m = arglist[1]
            n = arglist[2]
            k = arglist[3]
            dt = ToTorchType(EnumKernelDType(arglist[4]))
            self.init_cuda()
            self._init_AB_with_detailed_info(b,m,n,k,dt)
            self._init_torch_eps()
        
        while startFlag:
            if self.workFlag.value <= 0 : # when accepted EXIT msg, wait the last batch test complete
                startFlag = False
                print("[D] Deal Last batch of pkls!")
            pathLock.acquire()
            pklFiles = glob.glob(PathManager.pikle_dir() + f'/{self._devId}/*.pkl')
            if len(pklFiles) <= 0 :
                if endsignal.value > 0:
                    # end proc
                    pathLock.release()
                    break
                else:
                    pathLock.release()
                    time.sleep(2)
            else : 
                try:
                    if remoteTester is not None :
                        lps = []
                        rps = []
                        for pkl in pklFiles:
                            lps.append(pkl)
                            rps.append(pkl[0:pkl.rfind("/")])
                            infos = deserialize_from_file(pkl)
                            # send kernel file to remote
                            for (kpm,inConfig,packedKernel) in infos :
                                local_kernelpath = packedKernel.m_launcher.m_kernelLib.m_filePath
                                remoteTester.upload_file(local_kernelpath,local_kernelpath[0:local_kernelpath.rfind("/")])
                        # send pkl files and send OK message to remote tester
                        remoteTester.upload_files(lps,rps)
                    else:
                        for pkl in pklFiles:
                            arr = deserialize_from_file(pkl)
                            valid_kernels += arr
                        total_kernel_count += len(valid_kernels)
                    # DEBUG: 模拟进程crashed
                    # if total_kernel_count > 10 :
                    #     raise Exception('A Debug Exception')
                    # print(f"====== Glob .pkl files : {len(pklFiles)}, Valid Kernels : {len(valid_kernels)} ========")
                    for pkl in pklFiles:
                        os.remove(pkl)
                        # print(f"deleted: {pkl}")
                except Exception as e:
                    print(f"exception occur when deal {pkl}: {e}")
                pathLock.release()
            
            # When use remote perftester, skip local benchmark
            if remoteTester is not None :
                continue
            # execute benchmark at local
            perf_data = []

            for (kpm,inConfig,packedKernel) in valid_kernels :
                dyTorchCounter+=1
                if self.matA is None or self.matB is None :
                    self._init_AB(kpm,inConfig)
                perf_data.append(self._test_perf(kpm, inConfig, packedKernel, benchmarkCount, warmupCount))        
                if len(torchDynamicLogPath) > 0 and int(dyTorchCounter) % int(self.check_dynamic_torch_perf) == 0:
                    self.check_torch_dynamic_perf(torchDynamicLogPath, dyTorchCounter)
            valid_kernels.clear()
            self._getBestPerf(perf_data, topNum)
            if len(self.BestPerf) > 0 and outputPAth is not None :
                with open(outputPAth,mode='w') as f:
                    obj = self.jsonfyBestPerfs()
                    json.dump(obj,f,indent=4)
            if not startFlag :
                # the last batch of pkls test complete. set finishFlag to notify tcp controller to reply the perfrecord file path
                print("=== finishflag set")
                finishflag.value = 1

        # end signal triggered
        print(f"=====[ PerfTest on Device {self._devId} Finished ] =======")
        if remoteTester is not None and socket_client is not None: # use remote benchmark
                # notify remoteTester stop globing pkls, wait for perftest ends, finally get the location of benchmark result
                print("== Compile ends. send EXIT msg")
                remotepath = socket_client.send_and_wait("EXIT")
                print("== waiting for remote log downloads ... ")
                if remoteTester.download_file(str(PathManager.project_dir()), remotepath) :
                    _lp = str(PathManager.project_dir()) + '/' + remotepath.split('/')[-1]
                    print(f"=== remote benchmark result has been downloaded : {_lp}  ")
                else:
                    print(f"=== remote benchmark result download failed! ")
        if socket_client is not None:
            socket_client.stop()
        if self.controllerProc is not None :
            self.controllerProc.join()
        return 0
        
class SerialCompileTask :
    def _task_compile_kernel(self, kpm : KernelArgMatmul, index:int, deviceId:int, backendtype : EnumBackendType, arch : str) -> Tuple[KernelArgMatmul,UserInputs,CompiledKernel] :
        Print = print
        # compile kernel
        # Print("===== KCGCompiler ctor ========")
        kernelCompiler = KCGCompiler()
        _backend = 0
        if backendtype.value == EnumBackendType.CUDA.value :
            _backend = 1
        elif backendtype.value == EnumBackendType.HIP.value :
            _backend = 2
        else:
            assert False, f'invalid backendtype {backendtype}, Ty is {type(backendtype)}'
        kernelCompiler.set_platform(_backend,arch)
        # Print("===== call compileKernel(kpm)[0] ========")
        hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,shmBytes = kernelCompiler.compileKernel(kpm)[0] 
        # print(f"blockdims = {blockDimX,blockDimY,blockDimZ}")
        # print(f"griddims = {gridDimX,gridDimY,gridDimZ}")
        # Print("========= hsacoPath = ",hsacoPath)
        # Print("========= kernelName = ",kernelName)
        # print(f"==== backend is {backendtype}")
        inConfig = UserInputs(hsacoPath,kernelName,kpm, backendtype)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Matmul
        inConfig.shmBytes = shmBytes
        packedKernel = CompiledKernelFactory.getKernel(inConfig, deviceId)
        return (kpm,inConfig,packedKernel)  # 
  
    def setup_logger(logfile) -> logging.Logger:
        logging.basicConfig(filename=logfile, filemode='w+', level=logging.INFO)
        logger = logging.getLogger(logfile)
        return logger
    
    def compile_kernels(self, lock, kernelArg: KernelArgMatmul, lbs=0,ubs=-1,namePrefix='',deviceId=0, backendtype = EnumBackendType.HIP, arch = "906") -> List:
        # 读取 JSON 文件
        output_path = f"{PathManager.pikle_dir()}/{deviceId}/valid_kernels_{namePrefix}_{lbs}_{ubs}.pkl"
        valid_kernels = [] 
        if ubs < 0:
            lbs = 0; ubs =1
        for i in range(lbs,ubs) :
            kernelCfg = kernelArg
            r = self._task_compile_kernel(kernelCfg,i, deviceId,backendtype,arch)  
            valid_kernels.append(r)
        
        lock.acquire()
        serialize_to_file(output_path,valid_kernels)
        lock.release()



class ParallelTaskManager :
    ctx = multiprocessing.get_context('spawn')
    Process = ctx.Process
    def __init__(self, devids : List[int], total_cfg_count , tuningSpaceJson : str , perf_out_path : str, 
                 benchmarkcnt = 5, warmupcnt = 1, keepTopNum = 1, torchDynamicLogPath='',nTorchEpsInitTest=50,
                 atol= 1e-4, rtol=1e-4, remoteTestser : RemoteSSHConnect = None):
        self.locks = [] # ParallelTaskManager.ctx.Lock()
        self.compileProcs = []
        self.tuningSpaceJson = tuningSpaceJson
        self.CFG_COUNT = total_cfg_count
        self.task_groups = []
        self.m_totalKernels = []
        self.endSignal = ParallelTaskManager.ctx.Manager().Value(ctypes.c_int,0)
        self.finishflags = []
        self.perfTestFinalId = ParallelTaskManager.ctx.Manager().Value(ctypes.c_int,0)
        self.perf_out_path = perf_out_path
        self.perfProcMonitors = []
        self.nBenchMark = benchmarkcnt
        self.nWarmup = warmupcnt
        self.devIds = devids
        self.topNum = keepTopNum
        self.torchDynamicLogPath = torchDynamicLogPath
        self.nTorchEpsInitTest = nTorchEpsInitTest
        self.atol = atol
        self.rtol = rtol
        self.sender = remoteTestser  # run preftest on remote host
        for devid in self.devIds :
            lock = ParallelTaskManager.ctx.Lock()
            self.locks.append(lock)
    
    def __del__(self) :
        pass
        # for lk in self.locks :
        #     lk.release()
    
    @staticmethod
    def _innerCreateTesterProc(dev,lock,
        endSignal,
        finishflag,
        outfilename,
        nBenchMark,
        nWarmup,
        topNum,
        torchDynamicLogPath,
        nTorchEpsInitTest,atol,rtol,remotesender, isAsRemoteTester, baselineInitList) :
        
        baseInit = OperatorBaseArgs()
        if len(baselineInitList) > 0 :
            baseInit.operatorKind = EnumOperator.Matmul
            baseInit.argList = baselineInitList
        tester = PerfTester(dev,atol,rtol,nTorchEpsInitTest,baseInit)
        
        parsedBests = []
        try:
            if os.path.exists(outfilename):
                with open(outfilename) as f :
                    obj = json.load(f)
                    for cfg in obj['results'] :
                        kpm = KernelArgMatmul(0,0,0,0,1,1,1)
                        ktr = KernelTestResult(kpm)
                        ktr.parseFromJson(cfg)
                        parsedBests.append(ktr)
                        # tester.torch_eps = ktr.torch_elapseTimeMs
        except Exception :
            pass
        if len(parsedBests) > 0 :
            tester.BestPerf = parsedBests
        rc = tester.runPerfTests(lock,endSignal,finishflag,outfilename,nBenchMark, nWarmup, topNum,torchDynamicLogPath , nTorchEpsInitTest, remotesender,isAsRemoteTester)
        del tester; tester = None
        if rc == 0:
            # recv EXIT msg. return normally
            endSignal.value = 1
        
    @staticmethod
    def _perfMonitorFunc(devId, 
        lock,
        endSignal,
        finishflag,
        perf_out_path,
        nBenchMark,
        nWarmup,
        topNum,
        torchDynamicLogPath,
        nTorchEpsInitTest,atol,rtol, remotesender,isAsRemoteTester,initializerList) :
        perfLog = f"{perf_out_path}_card{devId}.json"
        worker = ParallelTaskManager.Process(
            target= ParallelTaskManager._innerCreateTesterProc, 
            args=(devId, lock, endSignal,finishflag ,perfLog,nBenchMark,nWarmup, topNum, torchDynamicLogPath, nTorchEpsInitTest,atol,rtol,remotesender,isAsRemoteTester,initializerList))
        worker.start()
        lastDeathTime = 0
        deadtime = 0
        minDeathDurationSeconds = 30
        while True:
            worker.join()
            if endSignal.value == 1 :  # 进程收到结束信号正常结束
                print(f"======= PerfTester {devId} Stopped OK ==========")
                return
            else:
                deadtime = time.time()
                if lastDeathTime == 0 or (deadtime - lastDeathTime) > minDeathDurationSeconds :
                    lastDeathTime = deadtime
                    print(f"======= [W] PerfTester {devId} crash. Restart it ==========")
                    del worker; worker = None
                    time.sleep(3)
                    worker = ParallelTaskManager.Process(target= ParallelTaskManager._innerCreateTesterProc, 
                        args=(devId, lock, endSignal,perfLog, nBenchMark, nWarmup, topNum, torchDynamicLogPath, nTorchEpsInitTest,atol,rtol,remotesender,isAsRemoteTester,initializerList))
                    worker.start()
                else:
                    print(f"======= [Fatal] PerfTester {devId} crash too frequently(<30s). No Restart! ==========")
                    return
    # @staticmethod
    # def _createBaselineInit(devid,batch,m,n,k,datatype) :
    #     baselineTester = PerfTester(devid,0.001,0.001,400)
    #     baselineTester.init_cuda()
    #     baselineTester._init_AB_with_detailed_info(batch,m,n,k,datatype)
    #     baselineTester._init_torch_eps()
        
    # @staticmethod
    # def init_baseline_matmul(batch,m,n,k, datatype : torch.dtype, devids : List[int], fProcess) :
    #     waitlist = []
    #     for devid in devids :
    #         p = fProcess(target= ParallelTaskManager._createBaselineInit, args=(devid,batch,m,n,k,datatype))
    #         waitlist.append(p)
    #         p.start()
    #     for p in waitlist :
    #         p.join()
    #     print(f"===== All baseline init OK ======")
        
    def _initPerfMonitors(self,isAsRemoteTester,initArgList) :
        for i in range(len(self.devIds)) :
            devid = self.devIds[i]
            lock = self.locks[i]
            finishflag = ParallelTaskManager.ctx.Manager().Value(ctypes.c_int,0)
            self.finishflags.append(finishflag)
            monitor = ParallelTaskManager.Process(target= ParallelTaskManager._perfMonitorFunc,
                args=(devid,
                    lock,
                    self.endSignal,
                    finishflag,
                    self.perf_out_path,
                    self.nBenchMark,
                    self.nWarmup,
                    self.topNum,
                    self.torchDynamicLogPath,
                    self.nTorchEpsInitTest,
                    self.atol,self.rtol,
                    self.sender,
                    isAsRemoteTester,
                    initArgList
                ))  # 创建perfTest守护进程。当perftest进程意外挂掉，由守护进程重启之
            monitor.start()
            self.perfProcMonitors.append(monitor)

    def _createCompileTask(self,func,*params) :
        p = ParallelTaskManager.Process(target = func, args = (*params,))
        p.start()
        self.compileProcs.append(p)
    
    def _waitAllCompilers(self) :
        for s in self.compileProcs :
            s.join()
        self.compileProcs.clear()
    
    ## 从json文件里读取 cfgs，转化为 List[KernelArgMatmul] 
    def _get_kernelargMatmul(self, cfgstr : int, tse : TuningSpaceEncoder_Matmul) -> KernelArgMatmul : 
        kw = ConfigKeywords    
        config = tse.decode(cfgstr)
        arg = KernelArgMatmul(config[kw.KEY_M],config[kw.KEY_N],config[kw.KEY_K],config[kw.KEY_BATCH] ,
                            EnumKernelDType(config[kw.KEY_DTYPE_A]), 
                            EnumKernelDType(config[kw.KEY_DTYPE_B]),
                            EnumKernelDType(config[kw.KEY_DTYPE_C]))
        arg.BLOCK_SIZE_M = config[kw.KEY_BLOCK_SIZE_M]
        arg.BLOCK_SIZE_N = config[kw.KEY_BLOCK_SIZE_N]
        arg.BLOCK_SIZE_K = config[kw.KEY_BLOCK_SIZE_K]
        arg.THREAD_SIZE_M = config[kw.KEY_THREAD_SIZE_M]
        arg.THREAD_SIZE_N = config[kw.KEY_THREAD_SIZE_N]
        arg.WARP_SIZE = config[kw.KEY_WARP_SIZE]
        arg.BLOCK_LAYOUT_M = config[kw.KEY_BLOCK_LAYOUT_M]
        arg.BLOCK_LAYOUT_N = config[kw.KEY_BLOCK_LAYOUT_N]
        arg.WARP_LAYOUT_M = config[kw.KEY_WARP_LAYOUT_M]
        arg.WARP_LAYOUT_N = config[kw.KEY_WARP_LAYOUT_N]
        arg.isATranspose = config[kw.KEY_IS_A_TRANSPOSE]
        arg.GLOB_LOAD_WIDTH_A = config[kw.KEY_GLOB_LOAD_WIDTH_A]
        arg.GLOB_LOAD_WIDTH_B = config[kw.KEY_GLOB_LOAD_WIDTH_B]
        arg.WARP_SCATTER_WIDTH_A = config[kw.KEY_WARP_SCATTER_WIDTH_A]
        arg.WARP_SCATTER_WIDTH_B = config[kw.KEY_WARP_SCATTER_WIDTH_B]
        arg.THREAD_SCATTER_WIDTH_A = config[kw.KEY_THREAD_SCATTER_WIDTH_A]
        arg.THREAD_SCATTER_WIDTH_B = config[kw.KEY_THREAD_SCATTER_WIDTH_B]
        arg.LOCAL_SPLIT_U = config[kw.KEY_LOCAL_SPLIT_U]
        arg.BLOCK_MAPPING = config[kw.KEY_BLOCK_MAPPING]
        arg.GLOB_STORE_WIDTH = config[kw.KEY_GLOB_STORE_WIDTH]
        arg.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
        arg.REG_PREFETCH = config[kw.KEY_REG_PREFETCH]
        arg.SHARED_PREFETCH = config[kw.KEY_SHARED_PREFETCH]
        arg.LOAD_CONTINUOUS = config[kw.KEY_LOAD_CONTINUOUS]
        arg.REDUCE_C_CONTINUOUS = config[kw.KEY_REDUCE_C_CONTINUOUS]
        return arg
    
    def run(self, backendtype : EnumBackendType, archInfo : str, maxProcess = 10, needCompile = True, needPerfTest = True, startFrom = 0, isAsRemoteTester = False) :
        try:
            procCount = 0
            dealed = startFrom
            print(f"=== start from cfg[{startFrom}] =====")
            # make sub pkl dirs for each visible card
            for devid in self.devIds :
                path = f"{PathManager.pikle_dir()}/{devid}"
                os.makedirs(path,exist_ok=True)
            # start perftest processes
            tse = None
            cfgstrs = []
            batch = None; m = None; n = None; k = None; dtype = None
            if needCompile and not isAsRemoteTester:
                with open(self.tuningSpaceJson) as f :
                    obj = json.load(f)
                    tse = TuningSpaceEncoder_Matmul(obj['template'])
                    cfgstrs = obj['cfgs']
                    batch = obj['template'][ConfigKeywords.KEY_BATCH][0]
                    m = obj['template'][ConfigKeywords.KEY_M][0]
                    n = obj['template'][ConfigKeywords.KEY_N][0]
                    k = obj['template'][ConfigKeywords.KEY_K][0]
                    dtype = obj['template'][ConfigKeywords.KEY_DTYPE_C][0]
                    
            if needPerfTest:
                init_arg_list = []
                if not isAsRemoteTester :
                    init_arg_list = [batch,m,n,k,dtype]
                else:
                    # when act as remotetestser, RunManager may upload serveral init_arg files corresponding to several gpu cards to us. This need to be considered in future
                    init_f = glob.glob(str(PathManager.default_cache_dir()) + f"/init_arg_*.json")
                    if len(init_f) > 0:
                        for file in init_f:
                            with open(file) as f:
                                o = json.load(f)
                                init_arg_list = [ o['b'],o['m'],o['n'],o['k'],o['dtype'] ]
                            os.remove(file)
                            print(f"[D] deleted initArgFile: {file}")
                print('============ start init perf monitors ==============')
                self._initPerfMonitors(isAsRemoteTester,init_arg_list)
            # start compiling processes
            if needCompile :
                with open(self.tuningSpaceJson) as f :
                    obj = json.load(f)
                    tse = TuningSpaceEncoder_Matmul(obj['template'])
                    cfgstrs = obj['cfgs']
                
                sct = SerialCompileTask()
                for i in range(startFrom,len(cfgstrs)) :
                    selectDevID = dealed % len(self.devIds)
                    # print(f"=========== Dealing : cfgstrs[{i}] ================")
                    config = self._get_kernelargMatmul(cfgstrs[i],tse)
                    self._createCompileTask(sct.compile_kernels,self.locks[selectDevID],config,i,i+1,'deepgen', self.devIds[selectDevID], backendtype, archInfo)
                    procCount += 1; dealed += 1
                    if procCount >= maxProcess or i == self.CFG_COUNT-1:
                        print(f"========= Wating for Compile tasks [{dealed}/{self.CFG_COUNT}]  ============")
                        self._waitAllCompilers()
                        procCount = 0
                print(f"========= All Compile tasks Finished [{self.CFG_COUNT}] ! Wait benchmark stop ============")
                self.endSignal.value = 1
                
        except Exception as e :
            print("[Deepgen Exception]",e)
            
        except KeyboardInterrupt as ki :
            print("[Deepgen Interrupt] User Keyboard Interrupt : Stop All ...")
            self._waitAllCompilers()
            self.endSignal.value = 1
        finally:
            # 处理完毕，发出结束信号，等待全部进程结束
            if needPerfTest :
                for p in self.perfProcMonitors :
                    p.join()
                    print("======== Perf monitors stopped ========")
            