import numpy as np
import cv2
import dlib
from imutils import face_utils
import time
import serial
import threading
from utils import load_cascade, open_webcam, open_serial
from module_eye_config import CASCADE_PATH, SHAPE_PREDICTOR_PATH, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, SERIAL_PORT, BAUD_RATE, FPS

arduino_response = ""
stop_thread = False

def read_from_arduino(ser):
    global arduino_response, stop_thread
    while not stop_thread:
        try:
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting).decode().strip()
                if response:
                    arduino_response = response
                    print(f"Arduino response: {response}")
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            break

def send_signal_to_arduino(ser):
    try:
        print("Sending SLEEP_TRUE to Arduino")
        ser.write("SLEEP_TRUE\n".encode())
        ser.flush()
    except serial.SerialException as e:
        print(f"Error writing to serial port: {e}")

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def main():
    global stop_thread
    try:
        ser = open_serial(SERIAL_PORT, BAUD_RATE)
    except IOError as e:
        print(e)
        return

    time.sleep(5)  # 시리얼 통신 안정화 대기

    # 아두이노로부터 데이터를 읽는 스레드 시작
    read_thread = threading.Thread(target=read_from_arduino, args=(ser,), daemon=True)
    read_thread.start()

    COUNTER = 0
    ALARM_ON = False

    detector = load_cascade(CASCADE_PATH)
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = open_webcam('/dev/video0')
    vs.stream.set(cv2.CAP_PROP_FPS, FPS)
    fps = vs.stream.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second set to: {fps}")

    while True:
        frame = vs.read()
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

            if (ear < EYE_AR_THRESH):
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        threading.Thread(target=send_signal_to_arduino, args=(ser,)).start()
                    cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    stop_thread = True  # 스레드 종료 플래그 설정
    read_thread.join()  # 스레드가 안전하게 종료되도록 대기

    cv2.destroyAllWindows()
    vs.stop()
    ser.close()

if __name__ == "__main__":
    main()
