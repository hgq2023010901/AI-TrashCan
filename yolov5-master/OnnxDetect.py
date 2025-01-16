import onnx
import torch
import numpy as np
import pandas as pd
import onnxruntime
from utils.dataloaders import LoadImages

ort_session = onnxruntime.InferenceSession('best3.onnx')

source = 'a.jpg'

dataset = LoadImages(source)
ort_inputs = {'images': dataset}

pred_logits = ort_session.run(None,ort_inputs)[0]