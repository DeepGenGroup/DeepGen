#! /home/xushilong/anaconda3/bin/python

import glob
import os
import random
import subprocess


def delete_files_in_directory(directory):
    # 确保目录存在
    if os.path.exists(directory) :
        if os.path.isdir(directory):
            # 遍历目录中的所有文件和子目录
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                # 如果是文件，删除它
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Deleted files in {directory}")
        if os.path.isfile(directory) :
            os.remove(directory)
            print(f"Deleted files in {directory}")
    else:
        print(f"The directory {directory} does not exist.")

for i in range(500):
    try:
        # dirs = glob.glob('/tmp/kcg_kernel-*')
        dirs = glob.glob('/tmp/compile-ptx-src-*')
        random.shuffle(dirs)
        for dir in dirs:
            delete_files_in_directory(dir)
            os.rmdir(dir)
        dirs.reverse()
        for dir in dirs:
            delete_files_in_directory(dir)
            os.rmdir(dir)
    except Exception:
        pass