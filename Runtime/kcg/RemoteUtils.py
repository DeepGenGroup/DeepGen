import socket   

import paramiko
from scp import SCPClient
from typing import List

class RemoteFileSender :
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


class MyTCPServer :
    def __init__(self,listenPort):
        self.server = None
        self.port = listenPort    
        self.conn = None    
    def listen(self, callbackFunc) :
        import socket
        if self.server is not None :
            return
        socket_server = socket.socket()
        socket_server.bind(("localhost", self.port))
        # 监听端口
        socket_server.listen(1)
        # 等待客户端连接，accept方法返回二元元组(连接对象, 客户端地址信息)
        print(f"服务端已开始监听，正在等待客户端连接...")
        self.conn, address = socket_server.accept()
        print(f"接收到了客户端的连接，客户端的信息：{address}")
        
    def recv(self) -> str:
        data: str = self.conn.recv(4).decode("UTF-8")
        return data
        
    def stop(self):
        self.conn.close()
        if self.server is not None:
            self.server.close()
        
class MyTCPClient :
    def __init__(self):
        self.socket_client = None
        
    def connect(self, destip, destport) :
        import socket
        if self.socket_client is None:
        # 创建socket对象
            socket_client = socket.socket()
            # 连接到服务器
            socket_client.connect((destip, destport))
    
    def send_and_wait(self,send_msg) -> str :
        self.socket_client.send(send_msg.encode("UTF-8"))
        # 接受消息
        recv_data = self.socket_client.recv(4).decode("UTF-8")    # 1024是缓冲区大小，一般就填1024， recv是阻塞式
        return recv_data
    
    def send(self,send_msg) :
        self.socket_client.send(send_msg.encode("UTF-8"))
    
    def stop(self) :
        self.socket_client.close()
    