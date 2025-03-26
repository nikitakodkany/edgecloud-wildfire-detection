import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession("wildfire_model.onnx")
input_name = sess.get_inputs()[0].name

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (128, 128))
input_tensor = np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32) / 255.0

pred = sess.run(None, {input_name: input_tensor})[0]
cv2.imwrite("output_mask.jpg", (pred[0, 0] > 0.5) * 255)
