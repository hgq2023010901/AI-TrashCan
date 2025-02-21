from utils.operation import YOLO,DrawRect
import cv2

def detect(onnx_path='ReqFile/yolo11s.onnx',img_path=r'ReqFile/bus.jpg',show=True):
    '''
    检测目标，返回目标所在坐标如：
    {'crop': [57, 390, 207, 882], 'classes': 'person'},...]
    :param onnx_path:onnx模型路径
    :param img:检测用的图片
    :param show:是否展示
    :return:
    '''
    yolo = YOLO(onnx_path=onnx_path)
    img_cv = cv2.imread(img_path)
    det_obj = yolo.decect(img_cv)

    # 结果
    print (det_obj)

    # 画框框
    if show:
        img_show = DrawRect(img_cv,det_obj)
        cv2.imshow('Image', img_show)
        cv2.waitKey(0)

if __name__ == "__main__":
    detect()