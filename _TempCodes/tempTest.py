import multiprocessing
import os
import torch

envname = 'TEST_ENV'

def test_proc() :
    print(f"subpproc env = {os.environ.get(envname)}")
    
def main() :
    ctx = multiprocessing.get_context('spawn')
    os.environ[envname] = "111"
    print(f"==== main proc : {os.environ.get(envname)}")
    p = ctx.Process(target=test_proc)
    p.start()
    p.join()
    print('====== main stopped ')
    return

if __name__ == '__main__':
    main()