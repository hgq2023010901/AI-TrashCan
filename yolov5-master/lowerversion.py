import onnx

model = onnx.load("best3.onnx")
print("ONNX VERSION: ",model.ir_version)
for opset_import in model.opset_import:
    if opset_import.domain == "":
        print("ONNX Opset Version:", opset_import.version)
    else:
        print("Domain:", opset_import.domain, "Opset Version:", opset_import.version)
# model.ir_version = 1
# onnx.save_model(model, r"best4.onnx")
