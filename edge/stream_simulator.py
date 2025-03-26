import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import psutil
from datetime import datetime

IMG_ARRAY_PATH = "data/processed/images/X_test.npy"
MODEL_PATH = "models/wildfire_model.onnx"
FPS = 2
VISUALIZE = True
PROFILE = True
TRACK_MEMORY = True

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"logs/inference_log_{timestamp}.txt"
log_file = open(log_file_path, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

latencies = []
memory_usages = []

def infer_mask(img, frame_num):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    start_time = time.perf_counter()
    pred = session.run(None, {input_name: img})[0]
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

    log_msg = f"Frame {frame_num:03d} | Inference Time: {latency_ms:.2f} ms"

    if TRACK_MEMORY:
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 ** 2)  # MB
        memory_usages.append(mem_used)
        log_msg += f" | Memory Used: {mem_used:.2f} MB"

    if PROFILE:
        log(log_msg)

    return (pred[0, 0] > 0.5).astype(np.uint8)

images = np.load(IMG_ARRAY_PATH)

for i, frame in enumerate(images):
    mask = infer_mask(frame, i + 1)

    overlay = frame.copy()
    overlay[mask == 1] = [255, 0, 0]

    if VISUALIZE:
        cv2.imshow("Original", frame)
        cv2.imshow("Predicted Fire Mask", mask * 255)
        cv2.imshow("Overlay", overlay)
        key = cv2.waitKey(int(1000 / FPS))
        if key & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

if latencies:
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    log(f"\n📊 Avg Inference Time: {avg_latency:.2f} ms ± {std_latency:.2f} ms")

if memory_usages:
    avg_mem = np.mean(memory_usages)
    log(f" Avg Memory Used: {avg_mem:.2f} MB")

log_file.close()
print(f"\nInference log saved to: {log_file_path}")
