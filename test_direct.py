import cv2
import pyrealsense2 as rs
import numpy as np
import math
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





pipe = rs.pipeline()
cfg = rs.config()
device = cfg.resolve(rs.pipeline_wrapper(pipe)).get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if str(device.get_info(rs.camera_info.product_line)) == 'L500':
    cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

try:
    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = rs.align(rs.stream.color).process(frames)
        color_frm = aligned_frames.get_color_frame()
        depth_frm = aligned_frames.get_depth_frame()

        if not color_frm or not depth_frm:
            continue

#        print(depth_frame.get_profile().format())
#        print(depth_frame.get_units())
        s = depth_frm.__getstate__()
        c= depth_frm.get_data()
        #print(c)
        print("frame found")
        coef_units_to_metres=depth_frm.get_units()
        image = np.asanyarray(color_frm.get_data())
        depth_d = np.asanyarray(depth_frm.get_data())*coef_units_to_metres

        crop_x = (image.shape[1] - image.shape[0]) // 2
        crop_y = 0
        crop_width = image.shape[0]
        crop_height = image.shape[0]

        image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        inputBlob = cv2.dnn.blobFromImage(image, inScaleFactor, (inWidth, inHeight), meanVal, False)
        net.setInput(inputBlob, "data")
        detection = net.forward("detection_out")

        detectionMat = detection[0, 0, :, :]

        confidenceThreshold = 0.8
        for i in range(detectionMat.shape[0]):
            confidence = detectionMat[i, 2]

            if confidence > confidenceThreshold:
                objectClass = int(detectionMat[i, 1])
                xLeftBottom = int(detectionMat[i, 3] * image.shape[1])
                yLeftBottom = int(detectionMat[i, 4] * image.shape[0])
                xRightTop = int(detectionMat[i, 5] * image.shape[1])
                yRightTop = int(detectionMat[i, 6] * image.shape[0])

                detected_square = (xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom)
                detected_square = (max(0, detected_square[0]), max(0, detected_square[1]), min(crop_width, detected_square[0] + detected_square[2]),
                          min(crop_height, detected_square[1] + detected_square[3]))

                depth_roi = depth_d[detected_square[1]:detected_square[1] + detected_square[3], detected_square[0]:detected_square[0] + detected_square[2]]
                if depth_roi.size > 0:
                    object_depth = np.mean(depth_roi)

                    dtext=formatter_distance_text(object_depth)

                    label = f"{classNames[objectClass]} {dtext[0]}"
                    label2 = f"{dtext[1]}"
                    cv2.rectangle(image, (detected_square[0], detected_square[1]), (detected_square[2], detected_square[3]), (0, 255, 0), 2)
                    cv2.rectangle(image, (detected_square[0], detected_square[1] - 50), (detected_square[0] + len(label) * FONT_WIDTH, detected_square[1]),
                                  (125, 125, 125), -1)
                    cv2.putText(image, label, (detected_square[0], detected_square[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.putText(image, label2, (detected_square[0], detected_square[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Display Image", image)

        if cv2.waitKey(1) >= 0:
            break
except Exception as e:
    print(str(e))
finally:
    pipe.stop()
    cv2.destroyAllWindows()