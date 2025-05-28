# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

import logging
from typing import List,Type,Tuple
from kcg.Utils import *
from kcg.Kernel import *
from kcg.Operators import matmul
import sys
import numpy as np
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
import traceback

# import pymlir

def g_getBaselineStoreFileName(devId) -> str:
    return PathManager().default_cache_dir() + f'/BenchmarkTorchEps_{devId}.log'

def g_getBaselinePklPath(devid) -> str :
    return PathManager().default_cache_dir() + f'/gemmBase_{devid}.pkl'

class KernelTestResult :
    def __init__(self, tuningArgs : TuningArgsInterface):
        self.tuningArgs = tuningArgs
        self.isCorrect = False
        self.acc = 0.0
        self.kcg_elapseTimeMs = 0.0
        self.torch_elapseTimeMs = 0.0
        self.diffRate = 0.0
        self.maxError = 0.0
        self.opType = EnumOperator.Invalid
    def __str__(self):
        return "{" + f"op : {self.opType}, correct : {self.isCorrect}, acc : {self.acc}, torchMs : {self.torch_elapseTimeMs}], config : {self.tuningArgs}" + "}"
    def jsonfy(self) -> Dict :
        obj = { "op" : self.opType,
                "correct" : self.isCorrect, 
                "acc" : self.acc,
                "torchMs" : self.torch_elapseTimeMs,
                "kcgMs" : self.kcg_elapseTimeMs,
                "config" : self.tuningArgs.jsonfy()
            }
        return obj
    
    def parseFromJson(self,jsonObj) :
        self.opType = jsonObj['op']
        self.isCorrect = jsonObj['correct']
        self.acc = jsonObj['acc']
        self.torch_elapseTimeMs = jsonObj['torchMs']
        self.kcg_elapseTimeMs = jsonObj['kcgMs']
        if self.opType == EnumOperator.Matmul :
            self.tuningArgs = matmul.MatmulTuningArgs()
        elif self.opType == EnumOperator.Attention :
            ...
        elif self.opType == EnumOperator.Poll :
            ...
        elif self.opType == EnumOperator.Convolution :
            ...
        else:
            assert False, f"Invalid opType {self.opType}"
        self.tuningArgs.assignWithJson(jsonObj['config'])


class PerfTester :
    def __init__(self,devId:int,atol:float,rtol:float,nTorchEpsInitTest = 50, OpInstance : OpInterface = None):
        self.inputTensors_baseline = []
        self.inputTensors_kcg = []
        self.outputTensor = None
        self.mat_kcg = None
        self.result_baseline = None
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
        self.OpInstance = OpInstance
    
    @staticmethod
    def _controllerRemote(workflag,finishflag,perflogPath : str) :
        server = MyTCPServer()
        print(f"[D] TCP server start listen on {server.port}")
        server.listen()
        msg = ""
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
    
    def inner_test_torch(self) -> Tuple[torch.Tensor,float]:
        self.result_baseline, eps = self.OpInstance.Test_baseline()
        return (self.result_baseline, eps)
    
    def _init_torch_eps(self) :
        self.result_baseline, tempEps = self.inner_test_torch()
        if not self._read_torch_eps_from_file() :
            eps_torch_list = []
            for i in range(0, self.nTorchEpsInitTestCount) :
                _, eps_torch = self.inner_test_torch()
                eps_torch_list.append(eps_torch)
                self.torch_eps = np.median(eps_torch_list)
        with open(self._torchEpsStoreFile,'w') as f :
            f.write(str(self.torch_eps))
        print(f'======== torch baseline on dev{self._devId} init OK! result written to {self._torchEpsStoreFile} ========= ')
    
    def _read_torch_eps_from_file(self) :
        try :
            if os.path.exists(self._torchEpsStoreFile):
                with open(self._torchEpsStoreFile,'r+') as f:
                    self.torch_eps = float(f.readline()) 
            else:
                return False
        except Exception as e:
            return False
        except IOError as e:
            return False
        return True
        
    def _test_perf(self, inConfig : KernelConfigs, packedKernel : CompiledKernel, 
                   benchmarkCount = 5, warmupCount = 1 ) -> KernelTestResult:
        # self.init_cuda()
        self.init_cuda()
        self.OpInstance.InitInputTensorsWithDatalist(self._devId)
        for mat in self.OpInstance.InputTensors_Benchmark :
            print('InputTensors_Benchmark shape : ', mat.shape)
        
        for mat in self.OpInstance.InputTensors_Baseline :
            print('InputTensors_Baseline shape : ', mat.shape)
        
        # self.OpTy.InitInputTensors(inConfig, self._devId)
        resultContainer = KernelTestResult(self.OpInstance.TuningArgs)
        packedKernel.setDevice(0)  # when __init__, env has been set to actual device id. set 0 here
        self.OpInstance.InitBaselineOutputTensor( self._devId)
        
        print(f"packed: blockdim ={packedKernel.m_launcher.m_kernelLib.m_blockDims} ")
        print(f"packed: griddim ={packedKernel.m_launcher.m_kernelLib.m_gridDims} ")
        print(f"packed: dev ={packedKernel.m_launcher.m_kernelLib.m_device} ")
        print(f"packed: shm ={packedKernel.m_launcher.m_kernelLib.m_shmSize} ")
        print(f"packed: m_launcherLibPath ={packedKernel.m_launcher.m_launcherLibPath} ")
        print(f"packed: m_filePath ={packedKernel.m_launcher.m_kernelLib.m_filePath} ")
        print(f"packed: backend = {packedKernel.m_launcher.m_kernelLib.m_backendType} ")
        print(f"packed: func = {packedKernel.m_launcher.m_kernelLib.m_kernelFuncName} ")
        # print(f"packed: func = {packedKernel.m_launcher.m_kernelLib.m_kernelInfo.m_function} ")
        
        
        # 计算torch的eps
        if self.torch_eps <= 0 or self.result_baseline is None:
            self._init_torch_eps()

        # warmup
        self.OpInstance.Test_warmup(self.outputTensor, packedKernel, warmupCount)
        # benchmark
        res = []
        for i in range(0,benchmarkCount) :
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # outputTensor = self.OpTy.GetBenchmarkOutputTensor( self._devId)
            outputTensor = torch.empty(1024,1024,dtype=torch.float32, device="cuda:7")
            print(f"out shape = {outputTensor.shape}")
            (resultTensor, eps) = self.OpInstance.Test_benchmark(packedKernel,outputTensor,start_event,end_event)
            res.append(eps)
            if self.mat_kcg is None :
                self.mat_kcg = resultTensor
        # print("c=",self.mat_kcg)

        if torch.allclose(self.mat_kcg, self.result_baseline, atol=self._atol, rtol=self._rtol):
            print('test correct!')
            resultContainer.isCorrect = True
            resultContainer.torch_elapseTimeMs = self.torch_eps
            resultContainer.kcg_elapseTimeMs = np.median(res)
            print(f"median time(ms) : Deepgen = {resultContainer.kcg_elapseTimeMs} , PyTorch= {resultContainer.torch_elapseTimeMs}")
            resultContainer.acc = resultContainer.torch_elapseTimeMs/resultContainer.kcg_elapseTimeMs
            print(f"speed up: {resultContainer.acc}")
        else:
            resultContainer.isCorrect = False
            diff,max_error= self._compare_with_error(self.result_baseline, self.mat_kcg)
            resultContainer.maxError = max_error
            outSize = 1
            for dim in self.outputTensor.shape :
                outSize = outSize * dim
            resultContainer.diffRate = diff/outSize
            print(f'test fail! maxerror={max_error}, diffrate={resultContainer.diffRate}')
        packedKernel.deleteBinary()
        return resultContainer
    
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
    
    def runPerfTests(self, runMode : EnumRunMode , pathLock, endsignal,finishflag ,outputPAth = None, benchmarkCount = 5, warmupCount = 1, topNum = 6, torchDynamicLogPath = '', nTorchEpsInitTest = 50, remoteTesterSSH : RemoteSSHConnect = None, isAsRemoteTester = False) : 
        # collect kernels from pkl         
        assert self.OpInstance is not None
        assert self.OpInstance.BaseArgs is not None
        assert self.OpInstance.TuningArgs is not None
        valid_kernels = [] # List[Tuple[KernelArgMatmul,UserInputs,CompiledKernel]]
        total_kernel_count = 0
        dyTorchCounter = 0
        startFlag = True
        if isAsRemoteTester :
            print("[D] _startController",flush=True)
            # wait "upload finish or EXIT" signal from remote
            self._startController(outputPAth,finishflag)
        socket_client = None
        print(f"[D] runMode={runMode}")
        if runMode.value == EnumRunMode.CallRemotePerftester.value and  remoteTesterSSH is not None :
            # use remote benchmark, connect remoteTester and send initializer args of different tasks
            if remoteTesterSSH.connect():
                print(f"connect remotePerfTester success : destip={remoteTesterSSH.host}")
                if self.OpInstance is not None and self.OpInstance.BaseArgs.operatorKind != EnumOperator.Invalid :
                    initargJsonPath = PathManager.default_cache_dir() + "/" + self.initArgJsonName
                    self.OpInstance.BaseArgs.dumpToJson(initargJsonPath)
                    remoteTesterSSH.upload_file(initargJsonPath, PathManager.default_cache_dir())
            socket_client = MyTCPClient()
            connected = False
            for i in range(8):
                connected = socket_client.connect(destip= remoteTesterSSH.host)
                if connected :
                    break
                time.sleep(5)
            if not connected :
                assert False, f"[Fatal] connect tcpserver failed : {remoteTesterSSH.host} "
            else:
                print(f"[I] connect tcpserver success! destip={remoteTesterSSH.host}")
        else:
            print('[D] run local benchmark')
            # run local benchmark
            # self.init_cuda()
            # wait init arg file upload to dir
            
            # while startFlag :
            #     argfile = glob.glob(PathManager.default_cache_dir() +"/"+ self.initArgJsonName)
            #     if len(argfile) <= 0:
            #         time.sleep(1)
            #     else:
            #         break
            # self.OpTy.BaseArgs.parseFromJsonfile(argfile[0])
            self.init_cuda()
            self.OpInstance.InitInputTensorsWithDatalist(self._devId)
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
                    if remoteTesterSSH is not None :
                        lps = []
                        rps = []
                        for pkl in pklFiles:
                            lps.append(pkl)
                            rps.append(pkl[0:pkl.rfind("/")])
                            infos = deserialize_from_file(pkl)
                            # send kernel file to remote
                            for (kpm,inConfig,packedKernel) in infos :
                                local_kernelpath = packedKernel.m_launcher.m_kernelLib.m_filePath
                                remoteTesterSSH.upload_file(local_kernelpath,local_kernelpath[0:local_kernelpath.rfind("/")])
                        # send pkl files and send OK message to remote tester
                        remoteTesterSSH.upload_files(lps,rps)
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
            if remoteTesterSSH is not None :
                continue
            # execute benchmark at local
            perf_data = []

            for (kpm,inConfig,packedKernel) in valid_kernels :
                dyTorchCounter+=1
                perf_data.append(self._test_perf( inConfig, packedKernel, benchmarkCount, warmupCount))        
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
        if runMode.value == EnumRunMode.CallRemotePerftester.value and remoteTesterSSH is not None and socket_client is not None: # use remote benchmark
                # notify remoteTester stop globing pkls, wait for perftest ends, finally get the location of benchmark result
                print("== Compile ends. send EXIT msg")
                remotepath = socket_client.send_and_wait("EXIT")
                print("== waiting for remote log downloads ... ")
                if len(remotepath) > 1 and remoteTesterSSH.download_file(str(PathManager.project_dir()), remotepath) :
                    _lp = str(PathManager.project_dir()) + '/' + remotepath.split('/')[-1]
                    print(f"=== remote benchmark result [{remotepath}] has been downloaded : {_lp}  ")
                else:
                    print(f"=== remote benchmark result [{remotepath}] download failed! ")
        if socket_client is not None:
            socket_client.stop()
        if self.controllerProc is not None :
            self.controllerProc.join()
        return 0
        
class SerialCompileTask :
    def _task_compile_kernel(self, op : OpInterface, deviceId:int, backendtype : EnumBackendType, arch : str) -> Tuple[TuningArgsInterface, KernelConfigs, CompiledKernel] :
        return op.Compile( deviceId, backendtype,arch)
    
    def compile_kernels(self, lock, op : OpInterface, lbs=0,ubs=-1,namePrefix='',deviceId=0, backendtype = EnumBackendType.HIP, arch = "906") -> List:
        # 读取 JSON 文件
        output_path = f"{PathManager.pikle_dir()}/{deviceId}/valid_kernels_{namePrefix}_{lbs}_{ubs}.pkl"
        valid_kernels = [] 
        if ubs < 0:
            lbs = 0; ubs =1
        for i in range(lbs,ubs) :
            kernelTupleInfo = self._task_compile_kernel(op,deviceId,backendtype,arch)  
            valid_kernels.append(kernelTupleInfo)
        
        lock.acquire()
        serialize_to_file(output_path,valid_kernels)
        lock.release()



class ParallelTaskManager :
    ctx = multiprocessing.get_context('spawn')
    Process = ctx.Process
    def __init__(self, runMode : EnumRunMode, devids : List[int], total_cfg_count , tuningSpaceJson : str , perf_out_path : str, 
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
        self.runMode = runMode
        for devid in self.devIds :
            lock = ParallelTaskManager.ctx.Lock()
            self.locks.append(lock)
        self.OpTy : Type[OpInterface] = None
        
        
    def setTargetOp(self,op : Type[OpInterface] ) :
        self.OpTy = op
    
    def __del__(self) :
        pass
        # for lk in self.locks :
        #     lk.release()
    
    @staticmethod
    def _innerCreateTesterProc(Op : Type[OpInterface], runMode : EnumRunMode,
        dev,
        lock,
        endSignal,
        finishflag,
        outfilename,
        nBenchMark,
        nWarmup,
        topNum,
        torchDynamicLogPath,
        nTorchEpsInitTest,atol,rtol,remotesender, isAsRemoteTester, baselineInitList, baseArgFile) :
        
        OpInstance = Op()
        if baseArgFile is not None and len(baseArgFile) > 0 :
            OpInstance.BaseArgs.parseFromJsonfile(baseArgFile)
        # baselineInitList = OpInstance.BaseArgs.getIntDatalist()
        if len(baselineInitList) > 0 :
            OpInstance.InitBaseArgs(baselineInitList)
            # OpType.operatorKind = EnumOperator.Matmul
            # OpType.values = baselineInitList
        tester = PerfTester(dev,atol,rtol,nTorchEpsInitTest,OpInstance)
        
        parsedBests = []
        try:
            if os.path.exists(outfilename):
                with open(outfilename) as f :
                    obj = json.load(f)
                    for cfg in obj['results'] :
                        ktr = KernelTestResult(OpInstance.TuningArgs)
                        ktr.parseFromJson(cfg)
                        parsedBests.append(ktr)
                        # tester.torch_eps = ktr.torch_elapseTimeMs
        except Exception :
            pass
        if len(parsedBests) > 0 :
            tester.BestPerf = parsedBests
        rc = tester.runPerfTests( runMode, lock,endSignal,finishflag,outfilename,nBenchMark, nWarmup, topNum,torchDynamicLogPath , nTorchEpsInitTest, remotesender,isAsRemoteTester)
        del tester; tester = None
        if rc == 0:
            # recv EXIT msg. return normally
            endSignal.value = 1
        
    @staticmethod
    def _perfMonitorFunc(Op : Type[OpInterface],
        runMode : EnumRunMode,
        devId, 
        lock,
        endSignal,
        finishflag,
        perf_out_path,
        nBenchMark,
        nWarmup,
        topNum,
        torchDynamicLogPath,
        nTorchEpsInitTest,atol,rtol, remotesender,isAsRemoteTester,initializerList, baseArgFile : str = "") :
        perfLog = f"{perf_out_path}_card{devId}.json"
        worker = ParallelTaskManager.Process(
            target= ParallelTaskManager._innerCreateTesterProc, 
            args=(Op,runMode, devId, lock, endSignal,finishflag ,perfLog,nBenchMark,nWarmup, topNum, torchDynamicLogPath, nTorchEpsInitTest,atol,rtol,remotesender,isAsRemoteTester,
                  initializerList, baseArgFile))
        worker.start()
        lastDeathTime = 0
        deadtime = 0
        minDeathDurationSeconds = 30
        while True:
            worker.join()
            if endSignal.value == 1 :  # 进程收到结束信号正常结束
                print(f">>>> ======= PerfTester {devId} Stopped OK ==========")
                return
            else:
                deadtime = time.time()
                if lastDeathTime == 0 or (deadtime - lastDeathTime) > minDeathDurationSeconds :
                    lastDeathTime = deadtime
                    print(f"======= [W] PerfTester {devId} crash. Restart it ==========")
                    del worker; worker = None
                    time.sleep(3)
                    worker = ParallelTaskManager.Process(target= ParallelTaskManager._innerCreateTesterProc, 
                        args=(Op,runMode, devId, lock, endSignal,perfLog, nBenchMark, nWarmup, topNum, torchDynamicLogPath, nTorchEpsInitTest,atol,rtol,remotesender,isAsRemoteTester,
                              initializerList, baseArgFile))
                    worker.start()
                else:
                    print(f"======= [Fatal] PerfTester {devId} crash too frequently(<30s). No Restart! ==========")
                    return
        
    def _initPerfMonitors(self,isAsRemoteTester,initArgList, baseArgFile) :
        for i in range(len(self.devIds)) :
            devid = self.devIds[i]
            lock = self.locks[i]
            finishflag = ParallelTaskManager.ctx.Manager().Value(ctypes.c_int,0)
            self.finishflags.append(finishflag)
            monitor = ParallelTaskManager.Process(target= ParallelTaskManager._perfMonitorFunc,
                args=(
                    self.OpTy,
                    self.runMode,
                    devid,
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
                    initArgList,
                    baseArgFile
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
    # def _get_kernelargMatmul(self, cfgstr : int, tse : TuningSpaceEncoder) -> MatmulTuningArgs : 
    #     kw = ConfigKeywords    
    #     config = tse.decode(cfgstr)
    #     arg = MatmulTuningArgs(config[kw.KEY_M],config[kw.KEY_N],config[kw.KEY_K],config[kw.KEY_BATCH] ,
    #                         EnumKernelDType(config[kw.KEY_DTYPE_A]), 
    #                         EnumKernelDType(config[kw.KEY_DTYPE_B]),
    #                         EnumKernelDType(config[kw.KEY_DTYPE_C]))
    #     arg.BLOCK_SIZE_M = config[kw.KEY_BLOCK_SIZE_M]
    #     arg.BLOCK_SIZE_N = config[kw.KEY_BLOCK_SIZE_N]
    #     arg.BLOCK_SIZE_K = config[kw.KEY_BLOCK_SIZE_K]
    #     arg.THREAD_SIZE_M = config[kw.KEY_THREAD_SIZE_M]
    #     arg.THREAD_SIZE_N = config[kw.KEY_THREAD_SIZE_N]
    #     arg.WARP_SIZE = config[kw.KEY_WARP_SIZE]
    #     arg.BLOCK_LAYOUT_M = config[kw.KEY_BLOCK_LAYOUT_M]
    #     arg.BLOCK_LAYOUT_N = config[kw.KEY_BLOCK_LAYOUT_N]
    #     arg.WARP_LAYOUT_M = config[kw.KEY_WARP_LAYOUT_M]
    #     arg.WARP_LAYOUT_N = config[kw.KEY_WARP_LAYOUT_N]
    #     arg.isATranspose = config[kw.KEY_IS_A_TRANSPOSE]
    #     arg.GLOB_LOAD_WIDTH_A = config[kw.KEY_GLOB_LOAD_WIDTH_A]
    #     arg.GLOB_LOAD_WIDTH_B = config[kw.KEY_GLOB_LOAD_WIDTH_B]
    #     arg.WARP_SCATTER_WIDTH_A = config[kw.KEY_WARP_SCATTER_WIDTH_A]
    #     arg.WARP_SCATTER_WIDTH_B = config[kw.KEY_WARP_SCATTER_WIDTH_B]
    #     arg.THREAD_SCATTER_WIDTH_A = config[kw.KEY_THREAD_SCATTER_WIDTH_A]
    #     arg.THREAD_SCATTER_WIDTH_B = config[kw.KEY_THREAD_SCATTER_WIDTH_B]
    #     arg.LOCAL_SPLIT_U = config[kw.KEY_LOCAL_SPLIT_U]
    #     arg.BLOCK_MAPPING = config[kw.KEY_BLOCK_MAPPING]
    #     arg.GLOB_STORE_WIDTH = config[kw.KEY_GLOB_STORE_WIDTH]
    #     arg.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
    #     arg.REG_PREFETCH = config[kw.KEY_REG_PREFETCH]
    #     arg.SHARED_PREFETCH = config[kw.KEY_SHARED_PREFETCH]
    #     arg.LOAD_CONTINUOUS = config[kw.KEY_LOAD_CONTINUOUS]
    #     arg.REDUCE_C_CONTINUOUS = config[kw.KEY_REDUCE_C_CONTINUOUS]
    #     return arg
    
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
            
            ############### Parse tuning space file ###################
            tse = None
            cfgstrs = []
            print(f'[D] needCompile={needCompile}, needPerfTest={needPerfTest},isAsRemoteTester={isAsRemoteTester},archInfo={archInfo}')
            OpInstance = self.OpTy()
            # batch = None; m = None; n = None; k = None; dtype = None
            if needCompile and not isAsRemoteTester:
                with open(self.tuningSpaceJson) as f :
                    obj = json.load(f)
                    tse = TuningSpaceEncoder(obj['template'])
                    cfgstrs = obj['cfgs']
                    OpInstance.BaseArgs.parseFromTemplateDict(obj['template'])
                    
            baseArgInitFileName = ""
            if needPerfTest:
                init_arg_list = []
                if not isAsRemoteTester :
                    # init_arg_list = [batch,m,n,k,dtype]
                    init_arg_list = OpInstance.BaseArgs.getIntDatalist()
                else:
                    # when act as remotetestser, RunManager may upload serveral init_arg files corresponding to several gpu cards to us. This need to be considered in future
                    print("== waiting for cache/init_arg_  files  ")
                    init_f = []
                    while len(init_f) <= 0:
                        init_f = glob.glob(str(PathManager.default_cache_dir()) + f"/init_arg_*.json")
                        if len(init_f) > 0:
                            for file in init_f:
                                # with open(file) as f:
                                #     o = json.load(f)
                                #     init_arg_list = [ o['b'],o['m'],o['n'],o['k'],o['dtype'] ]
                                # os.remove(file)
                                pathAfterMove = shutil.move(file, PathManager.tmp_dir())
                                print(f"[D] founded initArgFile: {pathAfterMove}")
                                if baseArgInitFileName is None :
                                    baseArgInitFileName = pathAfterMove
                                OpInstance.BaseArgs.parseFromJsonfile(pathAfterMove)
                            break
                        else:
                            time.sleep(3)
                print('============ start init perf monitors ==============')
                init_arg_list = OpInstance.BaseArgs.getIntDatalist()
                self._initPerfMonitors(isAsRemoteTester,init_arg_list, baseArgInitFileName)
            # start compiling processes
            if needCompile :
                print(f"[D] start compiling. self.CFG_COUNT={self.CFG_COUNT}=======")
                with open(self.tuningSpaceJson) as f :
                    obj = json.load(f)
                    tse = TuningSpaceEncoder(obj['template'])
                    cfgstrs = obj['cfgs']
                
                sct = SerialCompileTask()
                print("[D] sct ctor done ")
                for i in range(startFrom,len(cfgstrs)) :
                    selectDevID = dealed % len(self.devIds)
                    # print(f"=========== Dealing : cfgstrs[{i}] ================")
                    OpInstance.TuningArgs.assignWithEncoder(cfgstrs[i],tse)
                    self._createCompileTask(sct.compile_kernels,self.locks[selectDevID], OpInstance,i,i+1,'deepgen', self.devIds[selectDevID], backendtype, archInfo)
                    procCount += 1; dealed += 1
                    if procCount >= maxProcess or i == self.CFG_COUNT-1:
                        print(f"========= Wating for Compile tasks [{dealed}/{self.CFG_COUNT}]  ============",flush=True)
                        self._waitAllCompilers()
                        procCount = 0
                print(f"========= All Compile tasks Finished [{self.CFG_COUNT}] ! Wait benchmark stop ============")
                self.endSignal.value = 1
                
        except Exception as e :
            print("[Deepgen Exception]",e)
            msg = traceback.format_exc();print(msg)
            
            
        except KeyboardInterrupt as ki :
            print("[Deepgen Interrupt] User Keyboard Interrupt : Stop All ...")
            self._waitAllCompilers()
            self.endSignal.value = 1
        finally:
            # 处理完毕，发出结束信号，等待全部进程结束
            for p in self.perfProcMonitors :
                p.join()
                print("======== Perf monitors stopped ========")
            