from model import ImagePreProcessor, ONNXModel
import numpy as np
import argparse

def test_prediction(image_path, expected_class=None):
    print(f"Testing with Image: {image_path}")
    
    pre = ImagePreProcessor()
    model = ONNXModel()
    
    input_array = pre.preprocess(image_path)
    output = model.predict(input_array)
    predicted_class = int(np.argmax(output))
    
    print(f"Predicted Class ID: {predicted_class}")
    
    if expected_class is not None:
        if predicted_class == expected_class:
            print("Test Passed!")
        else:
            print(f"Test Failed. Expected {expected_class} but got {predicted_class}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--expected", type=int, help="Expected ID")
    args = parser.parse_args()
    
    test_prediction(args.image, args.expected)