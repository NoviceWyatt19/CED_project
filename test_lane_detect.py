import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import exposure
import warnings

#경고문 무시
warnings.filterwarnings('ignore')

# 캠 실행 및 동작여부 및 영상 캡쳐 저장용 변수 선언
cap = cv2.VideoCapture(0)
ret_lane, frame_lane = cap.read()

# 사진 크기 조정
frame_lane = cv2.resize(frame_lane, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

height ,width = frame_lane.shape[:2]
#print(frame_lane.shape[:2]) #체크용

# 경사도 고려
temp = frame_lane[220:height-12, :width, 2]
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2BGRA)) # noise와 처리 효유을 높이기 위해 흑백으로 처리한다

# thresholds 설정
th_h, th_l = (150, 255), (50, 160), (0, 255)
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)

# noise의 영향을 줄이기 위해 sobel filter 사용할 것이다.
