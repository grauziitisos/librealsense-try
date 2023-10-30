import socket
import numpy as np
import json
import cv2

# Server settings
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 123      # Port to listen on
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

def runDNN(color_mat):
    inputBlob = cv2.dnn.blobFromImage(color_mat, inScaleFactor, (inWidth, inHeight), meanVal, False)
    net.setInput(inputBlob, "data")
    detection = net.forward("detection_out")
    detectionMat = detection[0, 0, :, :]
    confidenceThreshold = 0.8
    answer = []
    for i in range(detectionMat.shape[0]):
        confidence = detectionMat[i, 2]
        if confidence > confidenceThreshold:
            objectClass = int(detectionMat[i, 1])
            xLeftBottom = int(detectionMat[i, 3] * color_mat.shape[1])
            yLeftBottom = int(detectionMat[i, 4] * color_mat.shape[0])
            xRightTop = int(detectionMat[i, 5] * color_mat.shape[1])
            yRightTop = int(detectionMat[i, 6] * color_mat.shape[0])
            answer.append((xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom, objectClass))
    return answer


# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections
server_socket.listen()

print(f"Server listening on {HOST}:{PORT}")

while True:
    # Accept a connection from a client
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    try:
        with client_socket, client_socket.makefile('rb') as r:
        # Receive the data from the client
            header = r.readline()
            if not header: break
            metadata = json.loads(header)
            serial_data = r.read(metadata['length'])
            color_mat = np.frombuffer(serial_data, dtype=metadata['type']).reshape(metadata['shape'])
           # print(color_mat)

            # Get the shape of the received array
            #shape = color_mat.shape

            # Send the shape back to the client as a string
            #response = f"Shape: {shape}"
            response = runDNN(color_mat)
            client_socket.send(str(response).encode())

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the client socket
        client_socket.close()