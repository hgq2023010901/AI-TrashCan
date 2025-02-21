import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.ttk import Style, Progressbar
from utils.operation import YOLO,DrawRect
import cv2
import onnxruntime
# from ffpyplayer.player import MediaPlayer
from pydub import AudioSegment
import simpleaudio as sa



def resize_img(img_pil,size):
    return img_pil.resize(size,Image.LANCZOS)

class MainPage:
    def __init__(self,root):
        self.root = root
        self.root.title('主页面')
        Waiting(self.root)
        global yolo
        global cap
        cap=cv2.VideoCapture(0)
        yolo = YOLO(onnx_path='ReqFile/yolo11s.onnx')
        # VideoPlayTk(self.root)
        # DetectWindow(self.root)
class Waiting:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1280x800")
        self.Page=tk.Frame(self.root)
        self.Page.pack(fill=tk.BOTH, expand=True)
        # 创建打开文件按钮
        self.open_button = tk.Button(self.Page, text='开始', command=self.start)
        self.open_button.place(relx=0.2,rely=0.3,relheight=0.4,relwidth=0.6)
    def start(self):
        self.Page.destroy()
        VideoPlayTk(self.root)
# 定义视频播放器类
class VideoPlayTk:
    # 初始化函数
    def __init__(self, root):
        self.root = root
        self.Page=tk.Frame(self.root)
        self.Page.pack(fill=tk.BOTH, expand=True)
        # 创建一个画布用于显示视频帧
        self.canvas = tk.Canvas(self.Page,bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # 创建打开文件按钮
        self.open_button = tk.Button(self.Page, text='开始', command=self.open_file)
        self.open_button.pack(side=tk.LEFT)
        # 初始化播放器和播放状态标志
        self.audio_player = None
        self.video_player= None
        self.is_paused = False
        self.is_stopped = False
        self.find_trash = False
        self.open_file()
    # 打开文件的函数
    def open_file(self):
        # file_path = filedialog.askopenfilename()  # 弹出文件选择对话框
        file_path = 'source/trash0.mp4'
        self.audio_player = None
        self.video_player = None
        self.is_paused = False
        self.is_stopped = False
        self.find_trash = False
        self.start_video(file_path)  # 开始播放选择的视频文件

    # 开始播放视频的函数
    def start_video(self, file_path):
        self.video_player = cv2.VideoCapture(file_path)
        audio = AudioSegment.from_file(file_path,format="mp4")
        self.audio_player=sa.play_buffer(audio.raw_data, num_channels=audio.channels,bytes_per_sample=audio.sample_width,sample_rate=audio.frame_rate)
        self.play_video()  # 开始播放视频
        self.slow_detect()

    # 播放视频的函数
    def play_video(self):
        if self.is_stopped:
            self.video_player = None  # 如果停止播放，则释放播放器资源
            return
        ret,frame= self.video_player.read()  # 获取下一帧和帧间隔
        if frame is None:
            self.root.after(10, self.open_file())  # 如果没有帧，则稍后再试
            return
        # 将帧图像转换为Tkinter PhotoImage并显示在画布上
        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将OpenCV图像转换为Pillow图像，然后转换为PhotoImage
        im = Image.fromarray(frm)
        im = im.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # 保持对PhotoImage的引用，防止被垃圾回收

        # 如果没有暂停，则继续播放下一帧

        if not self.is_paused:
            self.root.after(15, self.play_video)
    # 切换暂停状态的函数
    def video_stop(self):
        self.is_stopped=True
        self.audio_player.stop()
        self.Page.destroy()
        DetectWindow(self.root)
    def slow_detect(self):
        ret, img_cv = cap.read()
        det_obj = yolo.decect(img_cv)
        if det_obj != []:
            self.find_trash=True
            self.video_stop()
        print(det_obj,self.find_trash)
        #print('111')
        if self.find_trash == False:
            self.root.after(1000,self.slow_detect)
class DetectWindow:
    def __init__(self,root):
        self.root = root
        self.Page = tk.Frame(self.root, width=450, height=250)
        self.Page.pack(fill=tk.BOTH, expand=True)
        # 创建一个画布用于显示视频帧
        self.canvas = tk.Canvas(self.Page, bg='black')
        self.canvas.place(x=0,y=0,relheight=1,relwidth=0.7)
        # 创建打开文本
        self.res_label = tk.Label(self.Page, text='结束')
        self.res_label.place(relx=0.7,y=0,relheight=0.8,relwidth=0.3)
        # 返回按钮
        self.open_button = tk.Button(self.Page, text='返回', command=self.switch_window)
        self.open_button.place(relx=0.7,rely=0.8,relheight=0.2,relwidth=0.3)

        # 初始化播放器和播放状态标志
        self.res_text=[]
        self.is_stopped = False
        self.fast_detect()

    def show_res(self,img_res):
        if self.is_stopped:
            return
        frame = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        # 将OpenCV图像转换为Pillow图像，然后转换为PhotoImage
        im = Image.fromarray(frame)
        im = im.resize((self.canvas.winfo_width(),self.canvas.winfo_height()),Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo  # 保持对PhotoImage的引用，防止被垃圾回收

    def fast_detect(self):
        ret, img_cv = cap.read()
        det_obj = yolo.decect(img_cv)
        print(det_obj)
        img_res=DrawRect(img_cv,det_obj)
        self.show_res(img_res)
        self.root.after(100, self.fast_detect)
    def switch_window(self):
        self.Page.destroy()
        self.is_stopped=True
        VideoPlayTk(self.root)
    def check_detct(self):
        a=1
    def send_message(self):
        b=1
    def read_message(self):
        c=1

# 程序入口点
if __name__ == '__main__':
    root = tk.Tk()  # 创建Tkinter根窗口
    MainPage(root)  # 创建视频播放器应用
    root.mainloop()  # 进入Tkinter事件循环