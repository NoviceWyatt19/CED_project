import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # 이미지와 같은 크기의 검은색 마스크 생성
    match_mask_color = 255  # 흰색
    cv2.fillPoly(mask, vertices, match_mask_color)  # 다각형 영역을 흰색으로 채움
    masked_image = cv2.bitwise_and(img, mask)  # 원본 이미지와 마스크를 AND 연산
    return masked_image

def get_fitline(img, f_lines):
    try:
        lines = np.squeeze(f_lines)
        if len(lines.shape) != 1:
            lines = lines.reshape(lines.shape[0] * 2, 2)
            rows, cols = img.shape[:2]
            output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = output[0], output[1], output[2], output[3]

            # 선의 시작점과 끝점을 계산
            x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
            x2, y2 = int(((img.shape[0] / 2 + 70) - y) / vy * vx + x), int(img.shape[0] / 2 + 70)
            result = [x1, y1, x2, y2]
            return result
    except:
        return None

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)  # 선 그리기

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
    LANEWIDTH = 3.7  
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 그레이스케일 변환
    canny_image = cv2.Canny(gray_image, 90, 200)  # Canny 엣지 검출 (임계값 조정)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    
    # Hough 변환을 사용하여 선 검출
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,                    # 거리 해상도 (픽셀 단위)
        theta=np.pi / 180,        # 각도 해상도 (라디안 단위)
        threshold=100,            # 선 검출을 위한 최소 교차점 수
        lines=np.array([]),
        minLineLength=40,         # 선의 최소 길이
        maxLineGap=100            # 선을 연결할 수 있는 최대 간격
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

    image_with_lines = cv2.addWeighted(temp, 0.8, image, 1, 0.0)
    return image_with_lines, center

def main():
    # ser = serial.Serial("/dev/ttyACM0", 9600)
    car_cascade = cv2.CascadeClassifier('./cars.xml')

    if car_cascade.empty():
        print("Error loading cascade file for car detection.")
        return
    
    cap = cv2.VideoCapture('./lane_1.avi')

    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_cnt = 0
    update_rate = 5

    while True:
        if cap.isOpened():
            ret, lane_frame = cap.read()
            if ret:
                lane_frame, center = process_lane_detection(lane_frame, car_cascade)
                cv2.imshow("Lane Detection", lane_frame)
                if abs(center) > 1.5:
                    # ser.write("SLEEP_TRUE".encode()) # 졸음 신호 아두이노로 송신
                    print("he looks like sleep")
            else:
                cap.release()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_cnt += 1

    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()

if __name__ == "__main__":
    main()
