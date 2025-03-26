import socket
import threading
import time   

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
        self.work_directory = None
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
                print(f"RemotePerfTester[SSH] error: {self.host}:{self.port}",e)
                return False
            print("RemotePerfTester[SSH] connect OK!")
            return True

    def upload_files(self,local_path : List[str], remote_path : List[str]):
        assert(len(local_path) == len(remote_path))
        try:
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
        except Exception as e:
            print("[ScpUploadsError]",e)
            return False
        return True
                
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
            print("[SCP upload error]",e, f"{local_path} -> {remote_path}" )
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
            print("[SCP download error]",e, f"{remote_path} -> {local_path}")
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
        except Exception as e:
            print(e)
        return

DEFAULT_PORT = 18888
DEFAULT_TIMEOUT = 30
MSG_LEN = 512
SEPMARK = ';'

import socket

def find_available_port(host='0.0.0.0', port=0):
    """
    查找一个可用的端口（若port=0则由系统自动分配）
    返回: (可用端口号, socket对象)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许重用端口
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)   # 空闲60秒后开始探测
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # 每隔10秒探测一次
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)     # 最多探测3次
    try:
        sock.bind((host, port))
        actual_port = sock.getsockname()[1]  # 获取实际分配的端口
        return actual_port, sock
    except OSError as e:
        sock.close()
        raise RuntimeError(f"无法绑定端口: {e}")



class MyTCPServer :
    def __init__(self,listenPort = DEFAULT_PORT):
        self.server = None
        self.port = listenPort    
        self.conn = None    
        self._thread_listen = None
        
    def __del__(self) :
        self.stop()
        
    def listen(self) :
        try:
            import socket
            if self.server is not None :
                return True
            local_ip = get_local_ip()
            _ , self.server = find_available_port(local_ip,self.port)
            print(f"tcpserver localAddr = {local_ip}:{self.port}",flush=True)
            # 监听端口
            self.server.listen(1)
            # 等待客户端连接，accept方法返回二元元组(连接对象, 客户端地址信息)
            print(f"tcpserver start listen ...",flush=True)
            self._wait_accept()
            return True
        except Exception as e:
            print(e)
        except OSError as e:
            print(e)
        return False
    
    def _wait_accept(self) :
        if self.server is not None:
            self.conn, address = self.server.accept()
            print(f"tcpserver accept client : {address}",flush=True)
            self.reply("Server accepted client")
    
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
    def __init__(self,heartbeat_interval, bind_port = 0):
        self.sock = None
        self.heartbeat_interval = heartbeat_interval
        self.connected = False
        self._thread_hearbeat = None
        self._thread_recv = None
        self._ev_stop = threading.Event()
        self.port = bind_port 
    
    def __del__(self):
        self.stop()
    
    def connect_and_wait(self, destip, destport = DEFAULT_PORT) :
        try:
            if self.sock is None:
            # 创建socket对象
                local_ip = get_local_ip()
                tempPort, self.sock = find_available_port(local_ip,self.port)
                # 连接到服务器
            self.sock.connect((destip, destport))
            print(f"[I] tcpclient connect {destip}:{destport} success! ",flush=True)
            reply = self.sock.recv(MSG_LEN).decode("UTF-8") 
            print("[D] server reply :",reply,flush=True)
        except Exception as e :
            print("[W] tcpclient error : ",e,flush=True)
            return False
        # connect success. start heartbeat thread
        self.connected = True
        self._thread_hearbeat = threading.Thread(target=self._send_heartbeat, daemon=True,args=(destip,destport,))
        self._thread_hearbeat.start()
        return True
    
    def _send_heartbeat(self,destip,destport):
        """定期发送心跳包"""
        while self.connected and not self._ev_stop.is_set():
            try:
                # 发送心跳包（示例内容为0字节）
                self.sock.sendall(b'')
                print("Sent heartbeat")
            except (BrokenPipeError, ConnectionResetError, OSError):
                print("[D] Heartbeat failed, triggering reconnect",flush=True)
                self.connected = False
                self.sock.close()
                self.sock.connect((destip, destport))
                break
            time.sleep(self.heartbeat_interval)   

    def send_and_wait(self,send_msg,expected_msg = "") -> str :
        self.sock.send(send_msg.encode("UTF-8"))
        # 接受消息
        if len(expected_msg) > 0:
            while True:
                recv_data = self.sock.recv(MSG_LEN).decode("UTF-8")    # 1024是缓冲区大小，一般就填1024， recv是阻塞式
                if recv_data.find(expected_msg) >= 0:
                    return recv_data
                time.sleep(1)
        else:
            return self.sock.recv(MSG_LEN).decode("UTF-8")

    def send(self,send_msg) :
        self.sock.send(send_msg.encode("UTF-8"))
    
    def recv(self) :
        ret = self.sock.recv(MSG_LEN)
        return ret.decode("UTF-8")
    
    def stop(self) :
        if self._ev_stop is not None :
            self._ev_stop.set()
        if self.sock is not None:
            self.sock.close()
            self.sock = None
    