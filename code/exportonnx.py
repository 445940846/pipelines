import torch
from pipemodel import PIPEResNet18

height,width=200,200
model=PIPEResNet18()
model.load_state_dict(torch.load("PIPEResNet18.pt"))
model.eval()

dummy_input1=torch.randn(1,3,height,width)
torch.onnx.export(model,(dummy_input1),"PIPEResNet18.onnx",verbose=True)