from imutils.video import VideoStream
from imutils import face_utils
import numpy as np # 유클리드 거리
import imutils # opencv 보완
import time
import dlib # 안면인식
import cv2 # opencv

# 두 점 사이의 유클리드 거리를 계산하는 함수
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# 눈의 각도를 계산하여 눈이 감겼는지를 판단하는 함수
def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])  # 눈의 세로 길이
    B = euclidean_dist(eye[2], eye[4])  # 눈의 가로 길이
    C = euclidean_dist(eye[0], eye[3])  # 눈의 대각선 길이
    ear = (A + B) / (2.0 * C)  # 눈의 각도 계산
    return ear

# 얼굴 캐스케이드 파일 경로와 facial landmark predictor 파일 경로를 직접 설정
cascade_path = "./haarcascade_frontalface_default.xml"  # 얼굴 캐스케이드 파일의 경로
shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"  # facial landmark predictor 파일의 경로

# 눈의 각도가 임계값 이하이면 눈 감은 것으로 판단하며, 연속된 프레임 수를 기록
EYE_AR_THRESH = 0.28  # 눈의 각도 임계값
EYE_AR_CONSEC_FRAMES = 10  # 눈 감은 연속 프레임 수

# 프레임 카운터와 경고 상태를 초기화
COUNTER = 0  # 눈 감은 연속 프레임 수 카운터
ALARM_ON = False  # 경고 발생 여부

# 얼굴 감지기와 facial landmark predictor를 로드
print("[INFO] facial landmark predictor를 로드..")
detector = cv2.CascadeClassifier(cascade_path)  # 얼굴 감지기
predictor = dlib.shape_predictor(shape_predictor_path)  # facial landmark predictor

# 왼쪽 눈과 오른쪽 눈의 인덱스를 가져옴
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 비디오 스트림을 시작
print("[INFO] 비디오 스트림을 시작..")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# FPS 설정 (예: 20 FPS)
vs.stream.set(cv2.CAP_PROP_FPS, 30)
fps = vs.stream.get(cv2.CAP_PROP_FPS)
print(f"Frames per second set to: {fps}")

# 주요 루프를 시작
while True:
    # 프레임을 읽어
    frame = vs.read()
    # frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 감지
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        # 얼굴 영역을 가져와서 facial landmark를 예측
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 왼쪽 눈과 오른쪽 눈의 좌표를 추출
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # 눈의 각도를 계산
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 눈 주변 경계상자
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 눈의 각도가 임계값 이하이면 카운터를 증가
        if (ear < EYE_AR_THRESH):
            COUNTER += 1

            # 일정 프레임 수 이상으로 눈이 감겨있으면 경고를 표시
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("drowsy")
                    
                cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 눈이 감겨 있지 않으면 카운터를 초기화하고 경고를 해제
            COUNTER = 0
            ALARM_ON = False

        # 프레임에 눈의 각도를 표시
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 화면에 프레임을 표시
    cv2.imshow("Frame", frame)
    
    # 'q' 키를 누르면 루프를 종료
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

# 비디오 스트림을 중지
cv2.destroyAllWindows()
vs.stop()
