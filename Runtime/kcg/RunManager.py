import json
from typing import Dict,List,Tuple

class _KEY :
    workgroups = "workgroups"
    compiler = "compiler"
    ssh_addr = "ssh_addr"
    cwd = "cwd"
    tuning_config_relative_paths = "tuning_config_relative_paths"
    tuning_space_relative_paths = "tuning_space_relative_paths"
    perflog_prefix_list = "perflog_prefix_list"
    max_process_count  = "max_process_count"
    perf_tester = "perf_tester"
    devids = "devids"

class _Compiler :
    def __init__(self):
        self.sshHost = ""
        self.sshPort = ""
        self.cwd = ""
        self.tuning_config_relative_paths = ""
        self.tuning_space_relative_paths = ""
        self.perflog_prefix_list = ""
        self.max_process_count = ""


class _Benchmarker :
    def __init__(self):
        self.sshPort = ""
        self.sshHost = ""
        self.cwd = ""
        self.devIds = []

    
class _WorkGroup :
    def __init__(self):
        m_compiler = None
        m_perfTester = None
        m_isUseRemoteBenchmark = True
    
    # parse on element in json list("workgroups" elements), build  _Benchmarker & _Compiler
    def build(self, workgroupElement : Dict) :
        pass
    
    # start compiler and perftester :
    def start(self) :
        pass

class WorkgroupManager :
    def __init__(self, startupJson : str):
        self.m_config = None
        self.m_startupJsonPath = startupJson
    
    # build workgorups with startupJson
    def loadConfig(self):
        with open(self.m_startupJsonPath) as f :
            self.m_config = json.load(f)
        # check json format
        assert self.m_config is not None
    
