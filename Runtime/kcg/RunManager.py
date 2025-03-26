import json
from typing import Dict,List,Tuple
from RemoteUtils import DEFAULT_PORT, RemoteSSHConnect
from Utils import *

# 分发给各个机器的启动参数，用于被 main_processs() 读取
class StartParam :
    def __init__(self):
        self.tuning_param_file  = []
        self.perfPathPrefix = []
        self.cacheTuningSPaceFile = []
        self.maxCompilingProcess = 0
        self.gpu_devices = []
        self.tuningSpaceGenMode = 1
        self.backendType = EnumBackendType.INVALID
        self.arch = "80"
        self.benchmarkcnt = 10
        self.warmupcnt = 1
        self.keepTopNum = 100
        self.torchDynamicLogPath = ""
        self.nTorchEpsInitTest = 400
        self.atol = 1e-3
        self.rtol = 1e-3
        self.remoteTesterIP = ""
        self.remoteTesterSSHPort = 22
        self.remoteTesterUsername = ""
        self.remoteTesterPwd = ""
        self.remoteTesterCWD = ""
        self.runMode = EnumRunMode.GetTuneSpace_Local_Only
        self.tcp_port = DEFAULT_PORT
        self.start_from = 0
        
    def parseFromJson(self,path) :
        obj = None
        with open(path) as f:
            print(f"==== startParam parsing {path} ")
            obj = json.load(f)
        assert obj is not None
        
        self.tuning_param_file = obj['tuning_param_file']
        self.perfPathPrefix = obj['perfPathPrefix']
        self.cacheTuningSPaceFile = obj['cacheTuningSPaceFile']
        self.maxCompilingProcess = obj['maxCompilingProcess']
        self.gpu_devices = obj['gpu_devices']
        self.tuningSpaceGenMode = obj['tuningSpaceGenMode']
        if obj['backendType'] == "CUDA" :
            self.backendType = EnumBackendType.CUDA
        elif obj['backendType'] == "HIP" :
            self.backendType = EnumBackendType.HIP
        else:
            self.backendType = EnumBackendType.INVALID
            print(f"[Fatal] illegal backend Type {obj['backendType']}",flush=True)
            assert False , f"illegal backend Type {obj['backendType']}"
        self.arch = obj['arch']
        self.benchmarkcnt = obj['benchmarkcnt']
        self.warmupcnt = obj['warmupcnt']
        self.keepTopNum = obj['keepTopNum']
        self.torchDynamicLogPath = obj['torchDynamicLogPath']
        self.nTorchEpsInitTest = obj['nTorchEpsInitTest']
        self.atol = obj['atol']
        self.rtol = obj['rtol']
        self.remoteTesterIP = obj['remoteTesterIP']
        self.remoteTesterSSHPort = obj['remoteTesterSSHPort']
        self.remoteTesterUsername = obj['remoteTesterUsername']
        self.remoteTesterPwd = obj['remoteTesterPwd']
        if obj['runMode'] == "GetTuneSpace_Local_Only" :
            self.runMode = EnumRunMode.GetTuneSpace_Local_Only
        elif obj['runMode'] == "CallRemotePerftester" :
            self.runMode = EnumRunMode.CallRemotePerftester
        elif obj['runMode'] == "AsRemotePerftester" :
            self.runMode = EnumRunMode.AsRemotePerftester
        elif obj['runMode'] == "GetTuneSpace_Compile_Benchmark_Local" :
            self.runMode = EnumRunMode.GetTuneSpace_Compile_Benchmark_Local
        else:
            assert False, f"illegal runmode {obj['runMode']}"
        self.tcp_port = obj['tcp_port']
        self.remoteTesterCWD = obj['remote_tester_cwd']
        self.start_from = obj['start_from']
        
    def toJson(self) :
        dd = {
            'tuning_param_file' : None,
            'perfPathPrefix' : None,
            'cacheTuningSPaceFile' : None,
            'maxCompilingProcess' : None,
            'gpu_devices' : None,
            'tuningSpaceGenMode' : None,
            'backendType' : None,
            'arch' : None,
            'benchmarkcnt' : None,
            'warmupcnt' : None,
            'keepTopNum' : None,
            'torchDynamicLogPath' : None,
            'nTorchEpsInitTest' : None,
            'atol' : None,
            'rtol' : None,
            'remoteTesterIP' : None,
            'remoteTesterSSHPort' : None,
            'remoteTesterUsername' : None,
            'remoteTesterPwd' : None,
            'runMode' : None,
            'tcp_port' : None,
            'remote_tester_cwd' : None,
            'start_from' : None
        }
        dd['tuning_param_file'] = self.tuning_param_file
        dd['perfPathPrefix'] = self.perfPathPrefix
        dd['cacheTuningSPaceFile'] = self.cacheTuningSPaceFile
        dd['maxCompilingProcess'] = self.maxCompilingProcess
        dd['gpu_devices'] = self.gpu_devices
        dd['tuningSpaceGenMode'] = self.tuningSpaceGenMode
        dd['backendType'] = str(self.backendType)
        dd['arch'] = self.arch
        dd['benchmarkcnt'] = self.benchmarkcnt
        dd['warmupcnt'] = self.warmupcnt
        dd['keepTopNum'] = self.keepTopNum
        dd['torchDynamicLogPath'] = self.torchDynamicLogPath
        dd['nTorchEpsInitTest'] = self.nTorchEpsInitTest
        dd['atol'] = self.atol
        dd['rtol'] = self.rtol
        dd['remoteTesterIP'] = self.remoteTesterIP
        dd['remoteTesterSSHPort'] = self.remoteTesterSSHPort
        dd['remoteTesterUsername'] = self.remoteTesterUsername
        dd['remoteTesterPwd'] = self.remoteTesterPwd
        dd['runMode'] = str(self.runMode)
        dd['tcp_port'] = self.tcp_port
        dd['remote_tester_cwd'] = self.remoteTesterCWD
        dd['start_from'] = self.start_from
        return dd
        
class _Compiler :
    def __init__(self):
        self.ip_addr = ""
        self.sshPort = 0
        self.user_name = ""
        self.password = ""
        self.cwd = ""
        self.tuning_config_relative_paths = []
        self.tuning_space_relative_paths = []
        self.perflog_prefix_list = []
        self.max_process_count = 0
        self.tuning_space_generate_strategy = 1
        self.backendType = EnumBackendType.INVALID
        self.arch = ""
        self.start_from = 0
    
    def getUUID(self) -> str :
        return self.ip_addr +"_"+ self.user_name +"_"+ self.cwd
    
    def build(self, cfg : Dict) :
        self.ip_addr = cfg['ip_addr']
        self.sshPort =  cfg['ssh_port']
        self.user_name = cfg['user_name']
        self.password = cfg['password']
        self.cwd = cfg['cwd']
        self.tuning_config_relative_paths = cfg['tuning_config_relative_paths']
        self.tuning_space_relative_paths = cfg['tuning_space_relative_paths']
        self.perflog_prefix_list = cfg['perflog_prefix_list']
        
        if len(self.tuning_config_relative_paths) == len(self.tuning_space_relative_paths) \
            and len(self.tuning_space_relative_paths) == len(self.perflog_prefix_list) :
            pass
        else:
            assert False, f"[Fatal] Compiler {self.getUUID()} task list lengths illegal!"
        self.max_process_count = cfg['max_process_count']
        self.tuning_space_generate_strategy = cfg['tuning_space_generate_strategy']
        if cfg['backendType'] == 'CUDA':
            self.backendType = EnumBackendType.CUDA
        else:
            self.backendType = EnumBackendType.HIP
        self.arch = cfg['arch']
        self.start_from = cfg['start_from']
        
class _Benchmarker :
    def __init__(self):
        self.sshPort = 0
        self.ip_addr = ""
        self.cwd = ""
        self.devIds = []
        self.user_name = ""
        self.password = ""
        self.benchmark_count = 0
        self.warmup_count = 0
        
    def getUUID(self) -> str :
        ret = self.ip_addr +"_"+ self.user_name +"_"+ self.cwd
        for devid in self.devIds :
            ret += str(f'_dev{devid}')
        return ret
    
    def build(self,config : Dict) :
        self.ip_addr = config['ip_addr']
        self.sshPort = config['ssh_port']
        self.cwd = config['cwd']
        self.devIds = config['devids']
        self.user_name = config['user_name']
        self.password = config['password']
        self.benchmark_count = config['benchmark_count']
        self.warmup_count = config['warmup_count']
        self.keep_top = config['keep_top']
    
class _WorkGroup :
    def __init__(self,id = 0):
        self.m_compiler = None
        self.m_perfTester = None
        self.m_isUseRemoteBenchmark = True
        self.m_sshToCompiler = None
        self.m_sshToTester = None
        self.id = id
    # parse on element in json list("workgroups" elements), build  _Benchmarker & _Compiler
    def build(self, workgroupElement : Dict) :
        self.m_compiler = _Compiler()
        self.m_perfTester = _Benchmarker()
        self.m_compiler.build(workgroupElement['compiler'])
        self.m_perfTester.build(workgroupElement['perf_tester'])
        # When compiler and tester running on same ip, means that not use remote benchmark
        if self.m_compiler.ip_addr == self.m_perfTester.ip_addr :
            self.m_isUseRemoteBenchmark = False
    
    def getStartParamForCompilerAndTester(self) -> Tuple[StartParam,StartParam] :
        def __get_object() :
            com = StartParam()
            com.tuning_param_file = self.m_compiler.tuning_config_relative_paths
            com.perfPathPrefix = self.m_compiler.perflog_prefix_list
            com.cacheTuningSPaceFile = self.m_compiler.tuning_space_relative_paths
            com.maxCompilingProcess = self.m_compiler.max_process_count
            com.gpu_devices = self.m_perfTester.devIds
            com.tuningSpaceGenMode = self.m_compiler.tuning_space_generate_strategy
            com.backendType = self.m_compiler.backendType
            com.arch = self.m_compiler.arch
            com.benchmarkcnt = self.m_perfTester.benchmark_count
            com.warmupcnt = self.m_perfTester.warmup_count
            com.keepTopNum = self.m_perfTester.keep_top
            com.torchDynamicLogPath = ""
            com.nTorchEpsInitTest = 400
            com.atol = 1e-3
            com.rtol = 1e-3
            com.remoteTesterIP = self.m_perfTester.ip_addr
            com.remoteTesterSSHPort = self.m_perfTester.sshPort
            com.remoteTesterUsername = self.m_perfTester.user_name
            com.remoteTesterPwd = self.m_perfTester.password
            com.tcp_port = self.id + DEFAULT_PORT
            com.remoteTesterCWD = self.m_perfTester.cwd
            com.start_from = self.m_compiler.start_from
            return com
        com = __get_object()
        tester = __get_object()
        if self.m_isUseRemoteBenchmark :
            com.runMode = EnumRunMode.CallRemotePerftester
            tester.runMode = EnumRunMode.AsRemotePerftester
            return (com,tester)
        else:
            com.runMode = EnumRunMode.GetTuneSpace_Compile_Benchmark_Local
            return (com,com)
    
    def getCompilerTesterParamfileNames(self) -> Tuple[str,str] :
        c = PathManager.default_override_dir() + f"/param_compile_{self.id}.json"
        t = PathManager.default_override_dir() + f"/param_test_{self.id}.json"
        return (c,t)
    
    def __getStartCmd(self, wd : str ,shortfname : str) -> str :
        return f"cd {wd} ;nohup ./scripts/Benchmark.sh  {wd}/_cluster_run/{shortfname} &"
    def __getInitDirCmd(self, wd) -> str :
        return f"cd {wd} ; rm -rf ./cluster_run ; mkdir _cluster_run/"
    
    # start compiler and perftester :
    def start(self) :
        self.m_sshToTester = RemoteSSHConnect(
            self.m_perfTester.ip_addr,self.m_perfTester.sshPort,
            self.m_perfTester.user_name,self.m_perfTester.password)
        self.m_sshToCompiler = RemoteSSHConnect(
            self.m_compiler.ip_addr,self.m_compiler.sshPort,
            self.m_compiler.user_name,self.m_compiler.password)
        # generate json file locally. Then scp to remote
        param_c, param_t = self.getStartParamForCompilerAndTester()
        fname_c ,fname_t = self.getCompilerTesterParamfileNames()
        shortname_c = fname_c.split('/')[-1]
        shortname_t = fname_t.split('/')[-1]
        with open(fname_c, 'w') as f:
            json.dump(param_c.toJson(),f)
        with open(fname_t, 'w') as f:
            json.dump(param_t.toJson(),f)
        
        # connect to compiler and tester, execute startup shell command
        if self.m_sshToCompiler.connectSSH() and self.m_sshToTester.connectSSH() :
            
            self.m_sshToCompiler.execute_cmd_on_remote( self.__getInitDirCmd(self.m_compiler.cwd))
            self.m_sshToTester.execute_cmd_on_remote( self.__getInitDirCmd(self.m_perfTester.cwd))
            self.m_sshToCompiler.upload_file(fname_c,f"{self.m_compiler.cwd}/_cluster_run")
            self.m_sshToTester.upload_file(fname_t,f"{self.m_perfTester.cwd}/_cluster_run")
            self.m_sshToCompiler.execute_cmd_on_remote( self.__getStartCmd(self.m_compiler.cwd, shortname_c))
            self.m_sshToTester.execute_cmd_on_remote( self.__getStartCmd(self.m_perfTester.cwd, shortname_t))
        
        
# Wrokgroup 运行管理器，用于读取用户配置的批量信息 建立compile-test任务组 初始化各个机器的分工
# 一个Workgroup内的任务是串行执行的。
class WorkgroupManager :
    def __init__(self, startupJson : str):
        self.m_config = None
        self.m_startupJsonPath = startupJson
        self.workgroups : List[_WorkGroup] = []
        self.compilerUUIDs = []
        self.testerUUIDs = []
        
    # build workgorups with startupJson, then check whether all workgroups can run in parallel
    def _loadAndCheck(self) -> bool:
        with open(self.m_startupJsonPath) as f :
            self.m_config = json.load(f)
        # check json format
        assert self.m_config is not None
        
        for i in range(len(self.m_config['workgroups'])) :
            wg_param = self.m_config['workgroups'][i]
            wg = _WorkGroup(id=i)
            wg.build(wg_param)
            com_uid = wg.m_compiler.getUUID()
            tester_uid = wg.m_perfTester.getUUID()
            if com_uid not in self.compilerUUIDs:
                self.compilerUUIDs.append(com_uid)
            else:
                print(f"[E] workgroup {i} : compilerUUID {com_uid} already exsists!")  # deepGen 暂不支持多个compile任务同时运行在同台机器的相同项目目录
                return False
            if tester_uid not in self.testerUUIDs:
                self.testerUUIDs.append(tester_uid)
            else:
                print(f"[E] workgroup {i} : TesterUUID {com_uid} already exsists!")  # deepGen 暂不支持多个test任务同时运行在同台机器的相同项目目录
                return False
            self.workgroups.append(wg)
        return True
    
    def run(self) :
        if self._loadAndCheck() :
            for wg in self.workgroups :
                wg.start()
        