import socket   

import paramiko
from scp import SCPClient
from typing import List

class RemotePerfTester :
    def __init__(self,host,port,username,password):
        self.ssh = None
        self.host = host
        self.port = port
        self.username = username
        self.password = password
    def connect(self):
        if self.ssh is None :
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.ssh.connect(self.host, self.port, self.username, self.password,timeout=5)
            except Exception as e:
                print("RemoteFileSenderError[SSH] : ",e)

    def upload_files(self,local_path : List[str], remote_path : List[str]):
        assert(len(local_path) == len(remote_path))
        with SCPClient(self.ssh.get_transport()) as scp:
            for i in range(0,len(local_path)):
                lp = local_path[i]
                rp = remote_path[i]
                scp.put(lp, rp)
                
    def upload_file(self,local_path : str, remote_path : str):
        with SCPClient(self.ssh.get_transport()) as scp:
            scp.put(local_path, remote_path)

    def download_file(self,local_path : str, remote_path : str):
        try:
            with SCPClient(self.ssh.get_transport()) as scp:
                scp.get(remote_path,local_path)
        except Exception as e:
            print("[SCP error]",e)
            return False
        return True
    
DEFAULT_PORT = 18888
MSG_LEN = 512
SEPMARK = ';'

def get_local_ip() :
    import socket
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

class MyTCPServer :
    def __init__(self,listenPort = DEFAULT_PORT):
        self.server = None
        self.port = listenPort    
        self.conn = None    
    def listen(self) :
        import socket
        if self.server is not None :
            return
        self.server = socket.socket()
        local_ip = get_local_ip()
        print("local_ip=",local_ip,flush=True)
        self.server.bind((local_ip, self.port))
        # 监听端口
        self.server.listen(1)
        # 等待客户端连接，accept方法返回二元元组(连接对象, 客户端地址信息)
        print(f"服务端已开始监听，正在等待客户端连接...")
        self.conn, address = self.server.accept()
        print(f"接收到了客户端的连接，客户端的信息：{address}")
        
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
        self.conn.close()
        if self.server is not None:
            self.server.close()
        
class MyTCPClient :
    def __init__(self):
        self.socket_client = None
        
    def connect(self, destip, destport = DEFAULT_PORT) :
        try:
            if self.socket_client is None:
                import socket
            # 创建socket对象
                self.socket_client = socket.socket()
                # 连接到服务器
            self.socket_client.connect((destip, destport))
        except Exception as e :
            print("TCPClient Error : ",e)
            return False
        return True
        
    def send_and_wait(self,send_msg) -> str :
        self.socket_client.send(send_msg.encode("UTF-8"))
        # 接受消息
        recv_data = self.socket_client.recv(MSG_LEN).decode("UTF-8")    # 1024是缓冲区大小，一般就填1024， recv是阻塞式
        return recv_data
    
    def send(self,send_msg) :
        self.socket_client.send(send_msg.encode("UTF-8"))
    
    def stop(self) :
        self.socket_client.close()
    