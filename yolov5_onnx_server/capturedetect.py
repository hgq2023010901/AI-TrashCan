import cv2
from PIL import Image
import onnxruntime
import numpy as np
import time
from utils.orientation import non_max_suppression, tag_images
def detect(onnx_session,img_cv):
    img_size = 640
    img_size_h = img_size_w = img_size
    batch_size = 1
    num_classes = 4
    classes = ['Harmful', 'Kitchen Waste', 'Recyclable', 'Others']
    img = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))
    # to numpy
    def letterbox_image(image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
    resized = letterbox_image(img, (img_size_w, img_size_h))
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    image_numpy = img_in
    # -------

    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    # get input feed
    input_feed = {}
    for name in input_name:
        input_feed[name] = image_numpy

    #get output
    outputs = onnx_session.run(output_name, input_feed=input_feed)
    pred = non_max_suppression(outputs[0])

    if pred:
        res = tag_images(np.array(img), pred, img_size, classes)
    else:
        res = []
    return res

def DrawRect(img,det):
    img_res = img_cv
    for i in range(len(det)):
        img_res = cv2.rectangle(img_res, (det[i]['crop'][0], det[i]['crop'][1]),
                                (det[i]['crop'][2], det[i]['crop'][3]), (255, 0, 0), 2)
        img_res = cv2.putText(img_res, det[i]['classes'], (det[i]['crop'][0], det[i]['crop'][1]), font, 0.5,
                              (255, 255, 255), 1)


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    onnx_path = 'ReqFile/best3.onnx'
    onnx_sess = onnxruntime.InferenceSession(onnx_path)
    #onnx_session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider'])
    providers = onnx_sess.get_providers()
    print("available providers:",providers)
    device = onnxruntime.get_device()
    print("current device:",device)
    while True:
        t1 = time.perf_counter()
        ret , img_cv = video.read()
        det_obj = detect(onnx_sess,img_cv)
        img_res = DrawRect(img_cv,det_obj)
        t2 = time.perf_counter()
        print(det_obj,'fps:',1.00/(t2-t1))
        cv2.imshow('Image', img_cv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
