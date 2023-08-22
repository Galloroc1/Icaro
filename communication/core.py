import socket
import pickle


class Communicate:

    def __init__(self,
                 role,
                 name,
                 port,
                 other,
                 other_port):
        self.port = port
        self.role = role
        self.name = name
        self.other = other
        self.other_port = other_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send(self, dname, data, is_encrypt=False):
        self.sock.bind(('localhost', self.port))
        self.sock.listen(1)
        conn, addr = self.sock.accept()
        conn.sendall(self.encode_data(dname, data, is_encrypt))

    def encode_data(self, dname, data, is_encrypt=False):
        data_dict = {
            'dname': dname,
            'data': data,
            'is_encrypt': is_encrypt
        }
        data = pickle.dumps(data_dict)
        return data

    def get(self, dname):
        self.sock.connect(('localhost', self.other_port))
        packet = self.sock.recv(4096 * 100)
        msg = pickle.loads(packet)
        if msg["dname"] == dname:
            return {"data": msg['data'], "is_encrypt": msg['is_encrypt']}

    def close(self):
        self.sock.close()
