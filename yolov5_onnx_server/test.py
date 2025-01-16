import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.ttk import Style, Progressbar
from capturedetect import detect
import cv2
import onnxruntime
from ffpyplayer.player import MediaPlayer


# 定义视频播放器类
class VideoPlayTk:
    # 初始化函数
    def __init__(self, root):
        self.root = root
        self.root.title('视频播放器')  # 设置窗口标题

        # 创建一个画布用于显示视频帧
        self.canvas = tk.Canvas(root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 创建打开文件按钮
        self.open_button = tk.Button(root, text='开始', command=self.open_file)
        self.open_button.pack(side=tk.LEFT)

        # 初始化播放器和播放状态标志
        self.player = None
        self.is_paused = False
        self.is_stopped = False
        self.find_trash = False
        self.onnx_session = onnxruntime.InferenceSession('ReqFile/best3.onnx')
        self.cap = cv2.VideoCapture(0)
    # 打开文件的函数
    def open_file(self):
        # file_path = filedialog.askopenfilename()  # 弹出文件选择对话框
        file_path = 'source/trash0.mp4'
        self.player = None
        self.is_paused = False
        self.is_stopped = False
        self.find_trash = False
        self.start_video(file_path)  # 开始播放选择的视频文件

    # 开始播放视频的函数
    def start_video(self, file_path):
        self.player = MediaPlayer(file_path)  # 创建一个MediaPlayer对象
        self.play_video()  # 开始播放视频
        self.cap_detect()

    # 播放视频的函数
    def play_video(self):
        if self.is_stopped:
            self.player = None  # 如果停止播放，则释放播放器资源
            return

        frame, val = self.player.get_frame()  # 获取下一帧和帧间隔
        if val == 'eof':
            root.after(10, self.open_file())  # 如果到达视频末尾，则释放播放器资源
            return
        elif frame is None:
            self.root.after(10, self.play_video)  # 如果没有帧，则稍后再试
            return
        # 将帧图像转换为Tkinter PhotoImage并显示在画布上
        image, pts = frame
        image = Image.frombytes("RGB", image.get_size(), bytes(image.to_bytearray()[0]))
        root.winfo_width()
        root.winfo_height()
        image = image.resize((root.winfo_width(),
        root.winfo_height()-32),Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # 保持对PhotoImage的引用，防止被垃圾回收

        # 如果没有暂停，则继续播放下一帧
        if not self.is_paused:
            self.root.after(int(val * 1000), self.play_video)

    # 切换暂停状态的函数

    def video_stop(self):
        self.is_stopped=True
    def cap_detect(self):
        ret, img_cv = self.cap.read()
        det_obj = detect(self.onnx_session, img_cv)
        if det_obj != []:
            self.find_trash=True
            self.video_stop()
        print(det_obj,self.find_trash)
        #print('111')
        self.root.after(1000,self.cap_detect)

# 程序入口点
if __name__ == '__main__':
    root = tk.Tk()  # 创建Tkinter根窗口
    app = VideoPlayTk(root)  # 创建视频播放器应用
    root.mainloop()  # 进入Tkinter事件循环