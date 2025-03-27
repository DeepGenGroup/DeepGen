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
count = 0
for e1 in rr:
    for e2 in rr:
        count += 1
        # cmd = f"rm -rf /tmp/compile-ptx-src-{e}*.ptx"
        # print(cmd)
        # cmd = f"rm -rf /tmp/compile-ptx-src-{e}*.cubin"

        cmd1 = f"rm -rf /tmp/compile-ptx-log-{e1}{e2}* &"
        cmd2 = f"rm -rf /tmp/compile-ptx-src-{e1}{e2}* &"
        cmd3 = f"rm -rf /tmp/kcg_kernel-{e1}{e2}* &"
        # print(cmd1)
        # print(cmd2)
        print(cmd3)
    print("wait")
