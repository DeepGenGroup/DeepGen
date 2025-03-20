import socket   

import paramiko
from scp import SCPClient
from typing import List
import shutil

_local_ip = None

def get_local_ip():
    global _local_ip
    if _local_ip is not None:
        return _local_ip
    import socket
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        _local_ip = s.getsockname()[0]
    finally:
        s.close()
    return _local_ip

class RemoteSSHConnect :
    def __init__(self,destip,destport,username,password):
        self.ssh = None
        self.host = destip
        self.port = destport
        self.username = username
        self.password = password
        self._isLocalIP = False
        if self.host == get_local_ip():
            self._isLocalIP = True
    def __del__(self):
        if self.ssh is not None:
            self.ssh.close()

    def connectSSH(self):
        if self._isLocalIP :
            print("RemotePerfTester[SSH] connect OK(localhost, skip)")
            return True
        if self.ssh is None :
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.ssh.connect(self.host, self.port, self.username, self.password,timeout=5)
            except Exception as e:
                print("RemotePerfTester[SSH] error: ",e)
                return False
            print("RemotePerfTester[SSH] connect OK!")
            return True

    def upload_files(self,local_path : List[str], remote_path : List[str]):
        assert(len(local_path) == len(remote_path))
        if self._isLocalIP :
            for i in range(0,len(local_path)):
                lp = local_path[i]
                rp = remote_path[i]
                shutil.copy2(lp,rp)
        else:
            with SCPClient(self.ssh.get_transport()) as scp:
                for i in range(0,len(local_path)):
                    lp = local_path[i]
                    rp = remote_path[i]
                    scp.put(lp, rp)
                
    def upload_file(self,local_path : str, remote_path : str):
        try:
            if self._isLocalIP :
                if local_path != remote_path :
                    shutil.copy2(local_path,remote_path)
                else:
                    print(f'[W] SSH Upload: path same at local: {remote_path}, skip copy!')
            else:    
                with SCPClient(self.ssh.get_transport()) as scp:
                    scp.put(local_path, remote_path)
        except Exception as e :
            print("[SCP error]",e)
            return False
        return True
    
    def download_file(self,local_path : str, remote_path : str):
        if self._isLocalIP :
            if remote_path != local_path :
                shutil.copy2(remote_path,local_path)
            else:
                print(f'[W] SSH Download: path same at local: {remote_path}, skip copy!')
            return True
        try:
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.get(remote_path,local_path)
        except Exception as e:
            print("[SCP error]",e)
            return False
        return True
    
    def execute_cmd_on_remote(self,cmd:str) :
        try:
            print("== exec results ===",flush=True)
            if self._isLocalIP :
                import os
                os.system(cmd)
            else:
                myin, myout, myerr = self.ssh.exec_command(
                    cmd
                )
                for line in myout:
                    print(line,flush=True)
                for line in myerr:
                    print(line,flush=True)
        except ... as e:
            print(e)
        return

DEFAULT_PORT = 18888
MSG_LEN = 512
SEPMARK = ';'

class MyTCPServer :
    def __init__(self,listenPort = DEFAULT_PORT):
        self.server = None
        self.port = listenPort    
        self.conn = None    
        
    def __del__(self) :
        self.stop()
        
    def listen(self) :
        try:
            import socket
            if self.server is not None :
                return True
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            local_ip = get_local_ip()
            self.server.bind((local_ip, self.port))
            print(f"tcpserver localAddr = {local_ip}:{self.port}",flush=True)
            # 监听端口
            self.server.listen(1)
            # 等待客户端连接，accept方法返回二元元组(连接对象, 客户端地址信息)
            print(f"tcpserver start listen ...")
            self.conn, address = self.server.accept()
            print(f"tcpserver accept client : {address}")
            self.reply("Server accepted client")
            return True
        except Exception as e:
            print(e)
        except OSError as e:
            print(e)
        return False
        
    def recv(self) -> str:
        data: str = self.conn.recv(MSG_LEN).decode("UTF-8")
        return data
    
    def reply(self,data:str) :
        self.conn.send(data.encode("UTF-8"))
        
    def reply_and_wait(self,send_msg) -> str :
        self.conn.send(send_msg.encode("UTF-8"))
        # 接受消息
        recv_data = self.conn.recv(MSG_LEN).decode("UTF-8")    # 1024是缓冲区大小，一般就填1024， recv是阻塞式
        return recv_data
    
    def stop(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.server is not None:
            self.server.close()
            self.server = None
        
class MyTCPClient :
    def __init__(self):
        self.socket_client = None
    
    def __del__(self):
        self.stop()
    
    def connect(self, destip, destport = DEFAULT_PORT) :
        try:
            if self.socket_client is None:
                import socket
            # 创建socket对象
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 连接到服务器
            self.socket_client.connect((destip, destport))
        except Exception as e :
            print("[W] tcpclient error : ",e)
            return False
        print(f"[I] tcpclient connect {destip}:{destport} success! ")
        reply = self.socket_client.recv(MSG_LEN).decode("UTF-8") 
        print("[D] server reply :",reply)
        return True
        
    def send_and_wait(self,send_msg) -> str :
        self.socket_client.send(send_msg.encode("UTF-8"))
        # 接受消息
        recv_data = self.socket_client.recv(MSG_LEN).decode("UTF-8")    # 1024是缓冲区大小，一般就填1024， recv是阻塞式
        return recv_data
    
    def send(self,send_msg) :
        self.socket_client.send(send_msg.encode("UTF-8"))
    
    def stop(self) :
        if self.socket_client is not None:
            self.socket_client.close()
            self.socket_client = None
    