import cv2
import numpy as np
from collections import deque
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import time
import threading

# Lane Detection 관련 함수들
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # 단일 채널이므로 흰색으로 설정
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_region_of_interest(img, vertices):
    cv2.polylines(img, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

def get_fitline(img, f_lines):
    try:
        lines = np.squeeze(f_lines)
        if len(lines.shape) != 1:
            lines = lines.reshape(lines.shape[0] * 2, 2)
            rows, cols = img.shape[:2]
            output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = output[0][0], output[1][0], output[2][0], output[3][0]

            x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
            x2, y2 = int(((img.shape[0] / 2 + 70) - y) / vy * vx + x), int(img.shape[0] / 2 + 70)
            result = [x1, y1, x2, y2]
            return result
    except:
        return None

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
    if lines is not None:
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

def process_lane_detection(image, car_cascade, frame_cnt, check_rate=10):
    height, width = image.shape[0], image.shape[1]

    # 관심 영역 설정
    top_offset = 0.5  # 높이를 더 늘림
    bottom_extension = 0.08  # 기울기를 더 낮춤
    side_reduction = -0.16  # 가로 길이 더 줄임
    top_side_reduction = 0.05  # 윗변 가로 길이 더 줄임

    top_height = height * (1 - top_offset)
    bottom_height = height

    top_left = (width * (0.4 + side_reduction + top_side_reduction), top_height)
    top_right = (width * (0.6 - side_reduction - top_side_reduction), top_height)
    bottom_left = (width * (0.3 - bottom_extension + side_reduction), bottom_height)
    bottom_right = (width * (0.7 + bottom_extension - side_reduction), bottom_height)

    region_of_interest_vertices = [
        bottom_left,
        top_left,
        top_right,
        bottom_right
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Sobel 필터를 사용하여 x축과 y축으로 경계 검출
    sobelx = cv2.Sobel(blur_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(cv2.addWeighted(sobelx ** 2, 0.5, sobely ** 2, 0.5, 0))

    canny_image = cv2.Canny(np.uint8(sobel_combined), 40, 150)  # 상한값을 높임

    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # 관심 영역 시각화
    draw_region_of_interest(image, [np.array(region_of_interest_vertices, np.int32)])

    lines = cv2.HoughLinesP(cropped_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=10,
                            maxLineGap=70
                            )
    
    if lines is None:
        return image, 0

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
    center = 0

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

    # 차선 중심 계산 및 프레임 카운트에 따른 차선 이탈 여부 체크
    lane_departure = False
    if left_fit_line is not None and right_fit_line is not None:
        left_x1, left_y1, left_x2, left_y2 = left_fit_line
        right_x1, right_y1, right_x2, right_y2 = right_fit_line
        lane_center = (left_x1 + right_x1) // 2
        frame_center = width // 2

        # 차선 이탈 여부 체크
        offset_value = lane_center - frame_center
        if abs(offset_value) > 50:  # 임계값 조정 가능
            lane_departure = True

    image_with_lines = cv2.addWeighted(temp, 0.8, image, 1, 0.0)
    return image_with_lines, center, lane_departure

def lane_detection():
    car_cascade = cv2.CascadeClassifier('./cars.xml')

    if car_cascade.empty():
        print("Error loading cascade file for car detection.")
        return
    
    cap = cv2.VideoCapture('./change.avi')

    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_cnt = 0
    check_rate = 10  # 차선 이탈 여부를 체크하는 프레임 간격
    lane_departure_queue = deque(maxlen=50)  # 5초 동안의 데이터를 저장 (가정: 10FPS 비디오)

    while True:
        if cap.isOpened():
            ret, lane_frame = cap.read()
            if ret:
                lane_frame, center, lane_departure = process_lane_detection(lane_frame, car_cascade, frame_cnt, check_rate)
                
                if lane_frame is not None:
                    # cv2.imshow("Lane Detection", lane_frame)
                    pass
                else:
                    print("Received empty frame.")

                # 큐에 차선 이탈 여부 추가
                lane_departure_queue.append(lane_departure)

                # 큐의 70% 이상이 차선 이탈을 나타내면 경고 출력
                if sum(lane_departure_queue) / len(lane_departure_queue) > 0.7:
                    print("Significant Lane Departure Detected")

            else:
                cap.release()

        try:
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as e:
            print(f"Error waiting for key: {e}")
            break

        if key == ord("q"):
            break
        frame_cnt += 1

    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()

# Drowsiness Detection 관련 함수들
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def drowsiness_detection():
    CASCADE_PATH = "haarcascade_frontalface_default.xml"
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    EYE_AR_THRESH = 0.3
    FRAME_RATE = 30
    DROWSINESS_CHECK_SECONDS = 8
    DROWSINESS_CHECK_FRAMES = DROWSINESS_CHECK_SECONDS * FRAME_RATE
    DROWSINESS_THRESHOLD = 0.7

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame_cnt = 0
    update_rate = 5
    ear_display = 0

    COUNTER = 0
    QUEUE = deque(maxlen=DROWSINESS_CHECK_FRAMES)

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

            QUEUE.append(ear < EYE_AR_THRESH)
                         
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            
            else:
                COUNTER = 0

            if len(QUEUE) == DROWSINESS_CHECK_FRAMES:
                drowsiness_rate = sum(QUEUE) / len(QUEUE)
                if drowsiness_rate >= DROWSINESS_THRESHOLD:
                    print("Driver is drowsy!")
                    QUEUE.clear()

            if frame_cnt % update_rate == 0:
                ear_display = ear

            cv2.putText(frame, "EAR: {:.3f}".format(ear_display), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        try:
            # cv2.imshow("Frame", frame)
            pass
        except cv2.error as e:
            print(f"Error displaying frame: {e}")

        try:
            key = cv2.waitKey(33) & 0xFF  # 30 FPS를 위해 33ms 지연 시간 추가
        except cv2.error as e:
            print(f"Error waiting for key: {e}")
            break

        if key == ord("q"):
            break

        frame_cnt += 1

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    # 각각의 메인 함수를 별도의 스레드에서 실행
    lane_thread = threading.Thread(target=lane_detection)
    drowsiness_thread = threading.Thread(target=drowsiness_detection)

    # 스레드 시작
    drowsiness_thread.start()
    lane_thread.start()
    

    # 스레드가 종료될 때까지 대기
    lane_thread.join()
    drowsiness_thread.join()
