import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = image / 255.0  # Normalize to [0,1]
    return np.expand_dims(image.astype(np.float32), axis=0)

def infer_onnx(onnx_model_path, image_path):
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    image = preprocess_image(image_path)
    outputs = ort_session.run(None, {input_name: image})
    return outputs[0]

if __name__ == "__main__":
    onnx_model_path = "path/to/output_model.onnx"
    image_path = "path/to/test_image.jpg"
    output = infer_onnx(onnx_model_path, image_path)
    print("Model output:", output)
