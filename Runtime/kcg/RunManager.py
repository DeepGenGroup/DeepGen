import json
from typing import Dict,List,Tuple
from RemoteUtils import RemoteSSHConnect
# 集群运行管理器，用于读取用户配置的批量信息 建立compile-test任务集群 初始化各个机器的分工

class _Compiler :
    def __init__(self):
        self.ip_addr = ""
        self.sshPort = 0
        self.cwd = ""
        self.tuning_config_relative_paths = ""
        self.tuning_space_relative_paths = ""
        self.perflog_prefix_list = ""
        self.max_process_count = 0
        
    def build(self, cfg : Dict) :
        self.ip_addr = cfg['ip_addr']
        self.sshPort =  cfg['ssh_port']
        self.cwd = cfg['cwd']
        self.tuning_config_relative_paths = cfg['tuning_config_relative_paths']
        self.tuning_space_relative_paths = cfg['tuning_space_relative_paths']
        self.perflog_prefix_list = cfg['perflog_prefix_list']
        self.max_process_count = cfg['max_process_count']

class _Benchmarker :
    def __init__(self):
        self.sshPort = 0
        self.ip_addr = ""
        self.cwd = ""
        self.devIds = []
        self.user_name = ""
        self.password = ""
        
    def build(self,config : Dict) :
        self.ip_addr = config['ip_addr']
        self.sshPort = config['ssh_port']
        self.cwd = config['cwd']
        self.devIds = config['devids']
        self.user_name = config['user_name']
        self.password = config['password']
    
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
            self.m_sshToCompiler.execute_cmd_on_remote("")

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
