import socket
import json

ENCODING = 'ascii'
MAX_RESPONSE_SIZE_B = 1024
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
GET_STATUS = {"command": "get_status"}


class SocketClient:
    def __init__(self, host, port):
        print("Initializing manipulator connection on {}:{}".format(host, port))
        self.HOST = host
        self.PORT = port

    def send_coords_xyz(self, x, y, z):
        json_xyz = json.dumps({"x": x, "y": y, "z": z}).encode(encoding=ENCODING)
        return self.send_coords_obj(json_xyz)

    """Send coords object (encoded in bytes) to address and return the response in json"""
    def send_coords_obj(self, input):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(input)
            data = s.recv(MAX_RESPONSE_SIZE_B).decode(ENCODING)

        print(f"Received {data!r}")
        return json.dumps(data)

    """Request status of hardware to check if sending new command is appropriate"""

    def get_status(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data_bytes = bytes(json.dumps(GET_STATUS), ENCODING)
            s.sendall(data_bytes)
            data = s.recv(MAX_RESPONSE_SIZE_B)
            return json.dumps(data)

sc = SocketClient(HOST, PORT)
# json = json.dumps({"x":12.5,"y":0.1,"z":0})
# sc.send_coords_obj(json)
# sc.send_coords_obj("{\"x\":12.5,\"y\":0.1,\"z\":0}" )
print(sc.send_coords_xyz(3,2,1))