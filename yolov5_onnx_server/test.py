import onnxruntime as ort

session = ort.InferenceSession("ReqFile/yolo11s.onnx", providers=["CPUExecutionProvider"])

for input in session.get_inputs():
    print("input name: ", input.name)
    print("input shape: ", input.shape)
    print("input type: ", input.type)

for output in session.get_outputs():
    print("output name: ", output.name)
    print("output shape: ", output.shape)
    print("output type: ", output.type)

data = [[1, 2, 3], [4, 1, 6], [7, 8, 9]]
sorted_data = sorted(data, key=lambda x: x[1], reverse=True)


print(sorted_data)
"""
input name:  images
input shape:  [1, 3, 640, 640]
input type:  tensor(float)
output name:  output0
output shape:  [1, 25200, 9]
"""