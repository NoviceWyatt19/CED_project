# source venv/bin/activate
import cv2
import dlib
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import time
# import serial

# 상수 정의
EYE_AR_THRESH = 0.3  # 눈 깜빡임을 판단할 임계값
EYE_AR_CONSEC_FRAMES = 16  # 연속된 프레임 수 임계값
CASCADE_PATH = "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_fitline(img, f_lines):
    try:
        lines = np.squeeze(f_lines)
        if len(lines.shape) != 1:
            lines = lines.reshape(lines.shape[0] * 2, 2)
            rows, cols = img.shape[:2]
            output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = output[0], output[1], output[2], output[3]

            x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
            x2, y2 = int(((img.shape[0] / 2 + 70) - y) / vy * vx + x), int(img.shape[0] / 2 + 70)
            result = [x1, y1, x2, y2]
            return result
    except:
        return None

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def offset(left, mid, right):
    LANEWIDTH = 3.7  # 한국 도로 기준 [m]
    a = mid - left  # 도로 가운데 기준 왼쪽
    b = right - mid  # 도로 가운데 기준 오른쪽
    width = right - left

    if a >= b:  # 오른쪽으로 치우침
        offset = a / width * LANEWIDTH - LANEWIDTH / 2.0
    else:  # 왼쪽으로 치우침
        offset = LANEWIDTH / 2.0 - b / width * LANEWIDTH
    return offset

def process_lane_detection(image, car_cascade):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100)
    if lines is None:
        return image
    line_arr = np.squeeze(lines)
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    left_fit_line = get_fitline(temp, L_lines)
    right_fit_line = get_fitline(temp, R_lines)

    color = [255, 0, 0]

    if left_fit_line is not None and right_fit_line is not None:
        A = [left_fit_line[0], left_fit_line[1]]
        B = [left_fit_line[2], left_fit_line[3]]
        C = [right_fit_line[0], right_fit_line[1]]
        D = [right_fit_line[2], right_fit_line[3]]
        intersection = line_intersection((A, B), (C, D))

        car_mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(car_mask, [np.array([(intersection[0], 50), A, C], np.int32)], match_mask_color)

        car_masked_image = cv2.bitwise_and(image, car_mask)
        car_roi_gray = cv2.cvtColor(car_masked_image, cv2.COLOR_RGB2GRAY)
        cars = car_cascade.detectMultiScale(car_roi_gray, 1.4, 1, minSize=(80, 80))

        for (x, y, w, h) in cars:
            cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 255), 2)

        center = offset(left_fit_line[0], 180, right_fit_line[0])
        if abs(center) > 1.5:
            center_x = int(640 / 2.0)
            center_y = int(360 / 2.0)
            thickness = 2
            location = (center_x - 200, center_y - 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 3.5
            cv2.putText(temp, 'Warning', location, font, fontScale, (0, 0, 255), thickness)
            color = [0, 0, 255]

    if left_fit_line is not None:
        draw_fit_line(temp, left_fit_line, color)

    if right_fit_line is not None:
        draw_fit_line(temp, right_fit_line, color)

    image_with_lines = cv2.addWeighted(temp, 0.8, image, 1, 0.0)
    return image_with_lines

def main():
    # try:
    #     ser = serial.Serial("/dev/ttyACM0", 9600)
    # except serial.SerialException as e:
    #     print(f"Error opening serial port: {e}")
    #     return

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    car_cascade = cv2.CascadeClassifier('./cars.xml')

    if car_cascade.empty():
        print("Error loading cascade file for car detection.")
        return

    cap = cv2.VideoCapture('./change.avi')

    if not cap.isOpened():
        print("Error opening video file.")
        return

    vs = VideoStream(src=1).start()
    time.sleep(1.0)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame_cnt = 0
    update_rate = 5

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
                print("Eyes Closed!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 100), 2)
                # ser.write("SLEEP_TRUE".encode())  # 졸음 신호 아두이노로 송신

            if frame_cnt % update_rate == 0:
                ear_display = ear

            cv2.putText(frame, "EAR: {:.3f}".format(ear_display), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        if cap.isOpened():
            ret, lane_frame = cap.read()
            if ret:
                lane_frame = process_lane_detection(lane_frame, car_cascade)
                cv2.imshow("Lane Detection", lane_frame)
                # if abs(process_lane_detection.center) > 1.5:
                    # ser.write("SLEEP_TRUE".encode())  # 졸음 신호 아두이노로 송신
            else:
                cap.release()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    if cap.isOpened():
        cap.release()

if __name__ == "__main__":
    main()
