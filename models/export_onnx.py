import torch
from unet_model import UNet  
def export_to_onnx(model_path, onnx_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)  
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    model_path = "path/to/trained_model.pth"
    onnx_path = "path/to/output_model.onnx"
    export_to_onnx(model_path, onnx_path)
