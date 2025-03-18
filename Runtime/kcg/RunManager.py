import json
from typing import Dict,List,Tuple
from RemoteUtils import RemoteSSHConnect
from kcg.Utils import EnumBackendType
# 集群运行管理器，用于读取用户配置的批量信息 建立compile-test任务集群 初始化各个机器的分工

class StartParam :
    def __init__(self):
        self.tuning_param_file  = []
        self.perfPathPrefix = []
        self.cacheTuningSPaceFile = []
        self.maxCompilingProcess = 0
        self.gpu_devices = []
        self.tuningSpaceGenMode = 1
        self.backendType = EnumBackendType.CUDA
        self.arch = "80"
        self.benchmarkcnt = 10
        self.warmupcnt = 1
        self.keepTopNum = 100
        self.torchDynamicLogPath = ""
        self.nTorchEpsInitTest = 400
        self.atol = 1e-3
        self.rtol = 1e-3
        self.remoteTesterIP = ""
        self.remoteTesterSSHPort = ""
        self.remoteTesterUsername = ""
        self.remoteTesterPwd = ""
        
    def parseFromJson(self,path) :
        with open(path) as f:
            obj = json.load(f)
            self.tuning_param_file = obj['tuning_param_file']
            self.perfPathPrefix = obj['perfPathPrefix']
            self.cacheTuningSPaceFile = obj['cacheTuningSPaceFile']
            self.maxCompilingProcess = obj['maxCompilingProcess']
            self.gpu_devices = obj['gpu_devices']
            self.tuningSpaceGenMode = obj['tuningSpaceGenMode']
            self.backendType = obj['backendType']
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
    
class _Compiler :
    def __init__(self):
        self.ip_addr = ""
        self.sshPort = 0
        self.cwd = ""
        self.tuning_config_relative_paths = ""
        self.tuning_space_relative_paths = ""
        self.perflog_prefix_list = ""
        self.max_process_count = 0
        self.tuning_space_generate_strategy = 1
        self.backendType = EnumBackendType.INVALID
        self.arch = ""
        
    def build(self, cfg : Dict) :
        self.ip_addr = cfg['ip_addr']
        self.sshPort =  cfg['ssh_port']
        self.cwd = cfg['cwd']
        self.tuning_config_relative_paths = cfg['tuning_config_relative_paths']
        self.tuning_space_relative_paths = cfg['tuning_space_relative_paths']
        self.perflog_prefix_list = cfg['perflog_prefix_list']
        self.max_process_count = cfg['max_process_count']
        self.tuning_space_generate_strategy = cfg['tuning_space_generate_strategy']
        if cfg['backendType'] == 'CUDA':
            self.backendType = EnumBackendType.CUDA
        else:
            self.backendType = EnumBackendType.HIP
        self.arch = cfg['arch']
        
        
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
    def __init__(self):
        self.m_compiler = None
        self.m_perfTester = None
        self.m_isUseRemoteBenchmark = True
        self.m_sshToCompiler = None
        self.m_sshToTester = None
    
    # parse on element in json list("workgroups" elements), build  _Benchmarker & _Compiler
    def build(self, workgroupElement : Dict) :
        self.m_compiler = _Compiler()
        self.m_perfTester = _Benchmarker()
        self.m_compiler.build(workgroupElement['compiler'])
        self.m_perfTester.build(workgroupElement['perf_tester'])
        # When compiler and tester running on same ip, means that not use remote benchmark
        if self.m_compiler.ip_addr == self.m_perfTester.ip_addr :
            self.m_isUseRemoteBenchmark = False
    
    def getStartParam(self) -> StartParam :
        ret = StartParam()
        ret.tuning_param_file = self.m_compiler.tuning_config_relative_paths
        ret.perfPathPrefix = self.m_compiler.perflog_prefix_list
        ret.cacheTuningSPaceFile = self.m_compiler.tuning_space_relative_paths
        ret.maxCompilingProcess = self.m_compiler.max_process_count
        ret.gpu_devices = self.m_perfTester.devIds
        ret.tuningSpaceGenMode = self.m_compiler.tuning_space_generate_strategy
        ret.backendType = self.m_compiler.backendType
        ret.arch = self.m_compiler.arch
        ret.benchmarkcnt = self.m_perfTester.benchmark_count
        ret.warmupcnt = self.m_perfTester.warmup_count
        ret.keepTopNum = self.m_perfTester.keep_top
        ret.torchDynamicLogPath = ""
        ret.nTorchEpsInitTest = 400
        ret.atol = 1e-3
        ret.rtol = 1e-3
        ret.remoteTesterIP = self.m_perfTester.ip_addr
        ret.remoteTesterSSHPort = self.m_perfTester.sshPort
        ret.remoteTesterUsername = self.m_perfTester.user_name
        ret.remoteTesterPwd = self.m_perfTester.password
        return ret
    
    # start compiler and perftester :
    def start(self) :
        self.m_sshToTester = RemoteSSHConnect(
            self.m_perfTester.ip_addr,self.m_perfTester.sshPort,
            self.m_perfTester.user_name,self.m_perfTester.password)
        self.m_sshToCompiler = RemoteSSHConnect(
            self.m_compiler.ip_addr,self.m_compiler.sshPort,
            self.m_compiler.user_name,self.m_compiler.password)
        # connect to compiler and tester, execute startup shell command
        if self.m_sshToCompiler.connect() and self.m_sshToTester.connect() :
            self.m_sshToCompiler.execute_cmd_on_remote(f"cd {self.m_compiler.cwd} & . ./scripts/Benchmark.sh")
            self.m_sshToTester.execute_cmd_on_remote(f"cd {self.m_perfTester.cwd} & . ./scripts/Benchmark.sh")
        

class WorkgroupManager :
    def __init__(self, startupJson : str):
        self.m_config = None
        self.m_startupJsonPath = startupJson
        self.workgroups = []
    
    # build workgorups with startupJson
    def loadConfig(self):
        with open(self.m_startupJsonPath) as f :
            self.m_config = json.load(f)
        # check json format
        assert self.m_config is not None
        for wg in self.m_config['workgroups'] :
            temp = _WorkGroup()
            temp.build(wg)
            self.workgroups.append(temp)
