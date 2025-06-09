#! /home/xushilong/anaconda3/bin/python

import glob
import os
import random
import subprocess


def delete_files_in_directory(directory):
    # 确保目录存在
    if os.path.exists(directory) :
        if os.path.isfile(directory) :
            os.remove(directory)
            print(f"Deleted files in {directory}")
            return
        if os.path.isdir(directory):
            # 遍历目录中的所有文件和子目录
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                # 如果是文件，删除它
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Deleted files in {directory}")
            return
    else:
        print(f"The directory {directory} does not exist.")


rr = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,0,1,2,3,4,5,6,7,8,9'.split(',')
cc = 0
for e0 in rr:
    for e1 in rr:
        cmd = f"rm -rf /tmp/compile-ptx-src-{e0}{e1}*.ptx &\n" + f"rm -rf /tmp/compile-ptx-src-{e0}{e1}*.cubin &\n"
        if cc >= 40 :
            cc = 0
            cmd += "wait\n\n"
        else:
            cc += 1
        print(cmd)
print("wait")



# for i in range(500):
#     try:
#         # dirs = glob.glob('/tmp/kcg_kernel-*')
#         dirs = glob.glob('/tmp/compile-ptx-src-*')
#         random.shuffle(dirs)
#         for dir in dirs:
#             delete_files_in_directory(dir)
#             os.rmdir(dir)
#         dirs.reverse()
#         for dir in dirs:
#             delete_files_in_directory(dir)
#             os.rmdir(dir)
#     except Exception:
#         pass