import cv2
import pyrealsense2 as rs
import numpy as np
import math
from mjpeg_streamer import MjpegServer, Stream

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
FONT_WIDTH = 9

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

def formatter_distance_text(p):
    if(p>1):
        return [f"{math.floor(p)} m {math.floor((p-math.floor(p))*100)} cm", f"{((p*100-math.floor(p*100))*10):.4f} mm"]
    else:
        return [f"{math.floor((p-math.floor(p))*100)} cm", f"{((p*100-math.floor(p*100))*10):.4f} mm"]

stream = Stream("t", size=(640, 480), quality=100, fps=25)

#server = MjpegServer("10.42.0.159", 8080)
server = MjpegServer("78.154.137.242", 8080)
server.add_stream(stream)
server.start()



pipe = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipe)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(config)

try:
    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = rs.align(rs.stream.color).process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

#        print(depth_frame.get_profile().format())
#        print(depth_frame.get_units())
        coef_units_to_metres=depth_frame.get_units()
        color_mat = np.asanyarray(color_frame.get_data())
        depth_mat = np.asanyarray(depth_frame.get_data())*coef_units_to_metres

        crop_x = (color_mat.shape[1] - color_mat.shape[0]) // 2
        crop_y = 0
        crop_width = color_mat.shape[0]
        crop_height = color_mat.shape[0]

        color_mat = color_mat[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        inputBlob = cv2.dnn.blobFromImage(color_mat, inScaleFactor, (inWidth, inHeight), meanVal, False)
        net.setInput(inputBlob, "data")
        detection = net.forward("detection_out")

        detectionMat = detection[0, 0, :, :]

        confidenceThreshold = 0.8
        for i in range(detectionMat.shape[0]):
            confidence = detectionMat[i, 2]

            if confidence > confidenceThreshold:
                objectClass = int(detectionMat[i, 1])
                xLeftBottom = int(detectionMat[i, 3] * color_mat.shape[1])
                yLeftBottom = int(detectionMat[i, 4] * color_mat.shape[0])
                xRightTop = int(detectionMat[i, 5] * color_mat.shape[1])
                yRightTop = int(detectionMat[i, 6] * color_mat.shape[0])

                object = (xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom)
                object = (max(0, object[0]), max(0, object[1]), min(crop_width, object[0] + object[2]),
                          min(crop_height, object[1] + object[3]))

                depth_roi = depth_mat[object[1]:object[1] + object[3], object[0]:object[0] + object[2]]
                if depth_roi.size > 0:
                    object_depth = np.mean(depth_roi)

                    dtext=formatter_distance_text(object_depth)

                    label = f"{classNames[objectClass]} {dtext[0]}"
                    label2 = f"{dtext[1]}"
                    cv2.rectangle(color_mat, (object[0], object[1]), (object[2], object[3]), (0, 255, 0), 2)
                    cv2.rectangle(color_mat, (object[0], object[1] - 50), (object[0] + len(label) * FONT_WIDTH, object[1]),
                                  (125, 125, 125), -1)
                    cv2.putText(color_mat, label, (object[0], object[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.putText(color_mat, label2, (object[0], object[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        #cv2.imshow("Display Image", color_mat)
        #cv2.imshow(stream.name, color_mat)
        stream.set_frame(color_mat)
        if cv2.waitKey(1) >= 0:
            break
        
except Exception as e:
    print(str(e))
finally:
    pipe.stop()
    server.stop()
    cv2.destroyAllWindows()