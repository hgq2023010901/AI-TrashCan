from cv2 import waitKey

from capturedetect import detect
import tkinter as tk
from PIL import Image,ImageTk
import cv2

def test():
    print('abcd')
    root.after(1000, test)

def resize_img(img_pil,size):
    return img_pil.resize(size,Image.LANCZOS)

def update_frame():
    ret, frame = cap.read()
    if ret:
        # 将BGR颜色的OpenCV帧转换为RGB颜色
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将OpenCV图像转换为Pillow图像，然后转换为PhotoImage
        im = Image.fromarray(frame)
        im=resize_img(im,(800,450))
        img = ImageTk.PhotoImage(image=im)
        # 显示图像
        label.imgtk = img
        label.config(image=img)
        # 每隔10ms更新一次帧
        root.after(33, update_frame)
    else:
        # 如果视频结束，重新设置视频读取的起始位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        update_frame()  # 重新开始视频播放
        # 如果视频结束，释放捕捉对象
        # 下面这行用于 播放完就停止
        # cap.release()

if __name__ == "__main__":
    # 创建Tkinter窗口
    root = tk.Tk()
    root.title('视频播放器')
    root.wm_geometry('1280x800')
    # 创建一个标签用来显示视频帧
    label = tk.Label(root)
    label.pack()

    # 创建视频捕捉对象
    cap = cv2.VideoCapture('source/trash0.mp4')  # 替换为你的视频文件路径

    # 开始更新帧
    update_frame()
    test()
    # 运行应用
    print('222')
    root.mainloop()