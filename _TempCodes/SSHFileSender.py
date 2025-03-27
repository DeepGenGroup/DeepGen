# import os
# import paramiko
# import unicodedata
# from scp import SCPClient
 
# client = paramiko.SSHClient()
# client.load_system_host_keys()
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# client.connect('10.18.96.58', 2133, 'xushilong', 'xushilong')
# scp = SCPClient(client.get_transport())

# # 拿到服务器上所有文件夹


# print("== exec results ===",flush=True)
# myin, myout, myerr = client.exec_command(
#     ". /home/xushilong/DeepGen/Runtime/kcg/execHello.sh"
# )

# # 遍历远端服务器上的所有文件夹，若在本地服务器不存在，则scp过来
# for line in myout:
#     print(line)
# for line in myerr:
#     print(line)

# scp.close()
# client.close()

from multiprocessing import Process

def testfunc(testList) :
    for e in testList :
        print(e)
    print(len('/home/xushilong/DeepGen/_cache/loader_cuda.cpython-310-x86_64-linux-gnu.so'))
    import shutil
    shutil.copy2('/home/xushilong/DeepGen/Compile.sh','/home/xushilong/DeepGen/_dump/jjj.sh')

p = Process(target=testfunc,args=(["GEMM",1,2,3,4],))
p.start()
p.join()

