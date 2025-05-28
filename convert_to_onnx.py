import torch
from pytorch_model import Classifier, BasicBlock
import torchvision
import torch.onnx

model = Classifier(BasicBlock, [2, 2, 2, 2])
model.eval()

model.load_state_dict(torch.load("pytorch_model_weights.pth", map_location=torch.device("cpu")))

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ONNX model exported successfully as model.onnx")