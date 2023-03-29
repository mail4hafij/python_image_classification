import cv2  # Open source library for real time computer vision
import matplotlib.pyplot as plt

# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
config_file = 'cv2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'cv2/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)  # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))  # mobilenet => [-1, 1]
model.setInputSwapRB(True)  # bgr to rgb

# https://github.com/pjreddie/darknet/blob/master/data/coco.names
classLabels = []
file_name = 'cv2/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

"""
Object detection from image
"""
img = cv2.imread('images/car.jpg')  # bgr
classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for index, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (0, 0, 255), 2)
    cv2.putText(
        img,
        classLabels[index - 1],
        (boxes[0] + 10, boxes[1] + 40),
        font,
        fontScale=font_scale,
        color=(0, 255, 0),
        thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('images/output.jpg')