import torch
from ultralytics import YOLO

device = torch.device('cuda:0')
print('device', device)

model = YOLO('best3.pt')
path_out = model.export(format="onnx",opset=11)
print("path_out:",path_out)