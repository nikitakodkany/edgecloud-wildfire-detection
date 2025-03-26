import cv2
import os

images = sorted(os.listdir("stream_input/"))
for img_path in images:
    img = cv2.imread(f"stream_input/{img_path}")
    cv2.imshow("Stream", img)
    cv2.waitKey(500)  # simulate 2 FPS stream
