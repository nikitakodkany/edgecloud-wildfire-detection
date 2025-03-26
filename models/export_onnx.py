import torch
from unet import UNet

model = UNet()
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy_input, "wildfire_model.onnx", opset_version=11)
