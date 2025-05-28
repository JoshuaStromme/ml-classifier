import onnxruntime
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ImagePreProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0).numpy()
    
class ONNXModel:
    def __init__(self, model_path="model.onnx"):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    def predict(self, input_array):
        inputs = {self.session.get_inputs()[0].name: input_array}
        outputs = self.session.run(None, inputs)
        return outputs[0]
    
if __name__ == "__main__":
    pre = ImagePreProcessor()
    model = ONNXModel()
    
    input_array = pre.preprocess("n01440764_tench.jpeg")
    output = model.predict(input_array)
    
    predicted_class = int(np.argmax(output))
    print(f"Predicted Class ID: {predicted_class}")