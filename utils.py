import cv2
from imutils.video import VideoStream
import serial
import time

def load_cascade(cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError(f"Error loading cascade file {cascade_path}")
    return cascade

def open_webcam(video_path):
    cap = VideoStream(src=video_path).start()
    time.sleep(2)  # 비디오 스트림이 시작될 때까지 기다림
    frame = cap.read()
    if frame is None:
        raise IOError(f"Error opening video file {video_path}")
    return cap
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")
    return cap

def open_serial(port, baud_rate=9600):
    try:
        ser = serial.Serial(port, baud_rate)
    except serial.SerialException as e:
        raise IOError(f"Error opening serial port {port}: {e}")
    return ser

