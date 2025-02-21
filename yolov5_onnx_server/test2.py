from anyio import sleep
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa

audio = AudioSegment.from_file("trash0.mp4",format="mp4")
playback=sa.play_buffer(audio.raw_data, num_channels=audio.channels,bytes_per_sample=audio.sample_width,sample_rate=audio.frame_rate)
while True:
    sleep(0.01)
    print("aaa")