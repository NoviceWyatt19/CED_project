import cv2
import dlib
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import time
from collections import deque

# 상수 정의
EYE_AR_THRESH = 0.3  # 눈 깜빡임을 판단할 임계값
EYE_AR_CONSEC_FRAMES = 150  # 연속된 프레임 수 임계값 (30 FPS x 5초)

CASCADE_PATH = "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 두 점 사이의 유클리드 거리 계산 함수
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# 눈의 측면 비율(EAR) 계산 함수
def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def main():
    # ser = serial.Serial("/dev/ttyACM0", 9600)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame_cnt = 0
    update_rate = 5
    ear_display = 0

    # 졸음 여부 확정을 위한 queue 선언 및 초기화
    drowsy_queue = deque(maxlen=EYE_AR_CONSEC_FRAMES)
    drowsy = False

    while True:
        frame = vs.read()
        frame = imutils.resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                drowsy_queue.append(1)
                # ser.write("SLEEP_TRUE".encode()) # 졸음 신호 아두이노로 송신
            else:
                drowsy_queue.append(0)

            # 눈을 감고 있는 시간 판단
            if sum(drowsy_queue) >= EYE_AR_CONSEC_FRAMES * 0.8:  # 5초 동안 80% 이상의 프레임에서 눈을 감고 있다면
                drowsy = True
                print("sleep")
            else:
                drowsy = False

            # 눈을 뜨고 있는 경우 큐 초기화
            if not drowsy:
                drowsy_queue.clear()

            if frame_cnt % update_rate == 0:
                ear_display = ear

            cv2.putText(frame, "EAR: {:.3f}".format(ear_display), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, "Drowsy: {}".format(drowsy), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 100), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(33) & 0xFF  # 30 FPS를 위해 33ms 지연 시간 추가
        if key == ord("q"):
            break

        frame_cnt += 1

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
