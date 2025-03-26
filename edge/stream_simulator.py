import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import psutil
from datetime import datetime
import platform

# === CONFIG ===
IMG_ARRAY_PATH = "data/processed/images/X_test.npy"
MODEL_PATH = "models/wildfire_model.onnx"
FPS = 2  # Simulated streaming FPS
VISUALIZE = True
PROFILE = True
TRACK_MEMORY = True

# === SETUP LOGGING ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"inference_log_{timestamp}.txt"
log_file = open(log_file_path, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

# === LOAD MODEL ===
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
device = ort.get_device()

# === LOAD TEST IMAGES ===
images = np.load(IMG_ARRAY_PATH)
num_frames = len(images)

# === ENV INFO ===
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = os.path.basename(MODEL_PATH)

# === METRICS STORAGE ===
latencies = []
memory_usages = []

# === HEADER ===
log("Wildfire Detection Inference Log")
log(f"Model: {model_name}")
log(f"Dataset: CEMS Wildfire Dataset (Processed Test Set)")
log(f"Resolution: 128x128")
log(f"Total Frames: {num_frames}")
log(f"Device: {device} (ONNX Runtime)")
log(f"\n{'-'*60}")
log(f"{'Frame':<8}{'Inference Time (ms)':<24}{'Memory Usage (MB)' if TRACK_MEMORY else ''}")
log(f"{'-'*60}")

# === INFERENCE FUNCTION ===
def infer_mask(img, frame_num):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    start_time = time.perf_counter()
    pred = session.run(None, {input_name: img})[0]
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

    log_msg = f"{frame_num:<8}{latency_ms:<24.2f}"

    if TRACK_MEMORY:
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 ** 2)  # in MB
        memory_usages.append(mem_used)
        log_msg += f"{mem_used:.2f}"

    if PROFILE:
        log(log_msg)

    return (pred[0, 0] > 0.5).astype(np.uint8)

# === STREAMING SIMULATION ===
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

# === FINAL SUMMARY ===
log("-" * 60)
log(f"Average Inference Time: {np.mean(latencies):.2f} ms")
log(f"Standard Deviation:     {np.std(latencies):.2f} ms")
if memory_usages:
    log(f"Average Memory Used:    {np.mean(memory_usages):.2f} MB")
log(f"Log Timestamp: {run_timestamp}")

log_file.close()
print(f"\nLog saved to {log_file_path}")