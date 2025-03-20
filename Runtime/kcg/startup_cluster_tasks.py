from RunManager import *
import sys
import os

if __name__ == "__main__" :
    assert len(sys.argv) > 1
    clusterTaskConfig = sys.argv[1]
    clusterTaskConfig = str(os.path.realpath(clusterTaskConfig))
    PathManager.init()
    assert os.path.exists(clusterTaskConfig) 
    wgm =  WorkgroupManager(clusterTaskConfig)
    wgm.loadAndStart()
