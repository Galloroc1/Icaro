import socket
import pickle
from config import Config


class Communicate:

    COM_MAP = Config.COM_MAP
    NAME = Config.NAME
    ROLE = Config.ROLE
    PORT = Config.COM_MAP[NAME]

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send(self, who, dname, data, is_encrypt=False):
        self.sock.bind(('localhost', self.PORT))
        self.sock.listen(1)
        conn, addr = self.sock.accept()
        conn.sendall(self.encode_data(who, dname, data,is_encrypt))
        self.close()

    def encode_data(self, who, dname, data,is_encrypt=False):
        data_dict = {
            'who': who,
            'dname': dname,
            'data': data,
            'is_encrypt':is_encrypt
        }
        data = pickle.dumps(data_dict)
        return data

    def get(self, who, dname):
        self.sock.connect(('localhost', self.COM_MAP[who]))
        packet = self.sock.recv(4096*100)
        msg = pickle.loads(packet)
        self.close()
        if msg['who'] == self.NAME and msg["dname"] == dname:
            return {"data":msg['data'], "is_encrypt":msg['is_encrypt']}

    def close(self):
        self.sock.close()