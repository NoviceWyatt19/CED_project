import cv2
import numpy as np
from module_lane_img_processing import region_of_interest, draw_region_of_interest, get_fitline, draw_fit_line

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
    left_offset = mid - left
    right_offset = right - mid
    lane_width = right - left

    if left_offset >= right_offset:
        offset_value = left_offset / lane_width * LANEWIDTH - LANEWIDTH / 2.0
    else:
        offset_value = LANEWIDTH / 2.0 - right_offset / lane_width * LANEWIDTH
    return offset_value

def process_lane_detection(image, car_cascade, frame_count, check_rate=10, lane_departure_detected=False):
    height, width = image.shape[0], image.shape[1]
    top_offset_ratio = 0.5
    top_height = height * (1 - top_offset_ratio)
    bottom_height = height
    top_left = (width * (0.3), top_height)
    top_right = (width * (0.7), top_height)
    bottom_left = (width * (0.07), bottom_height)
    bottom_right = (width * (0.95), bottom_height)
    region_of_interest_vertices = [bottom_left, top_left, top_right, bottom_right]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(cv2.addWeighted(sobel_x ** 2, 0.5, sobel_y ** 2, 0.5, 0))
    canny_image = cv2.Canny(np.uint8(sobel_combined), 40, 150)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    draw_region_of_interest(image, [np.array(region_of_interest_vertices, np.int32)])
    lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=70)

    if lines is None:
        return image, 0, lane_departure_detected

    line_arr = np.squeeze(lines)
    slope_degrees = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi
    line_arr = line_arr[np.abs(slope_degrees) < 160]
    slope_degrees = slope_degrees[np.abs(slope_degrees) < 160]
    line_arr = line_arr[np.abs(slope_degrees) > 95]
    slope_degrees = slope_degrees[np.abs(slope_degrees) > 95]
    left_lines, right_lines = line_arr[(slope_degrees > 0), :], line_arr[(slope_degrees < 0), :]
    temp_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    left_lines, right_lines = left_lines[:, None], right_lines[:, None]

    left_fit_line = get_fitline(temp_image, left_lines)
    right_fit_line = get_fitline(temp_image, right_lines)

    line_color = [255, 0, 0]
    center = 0

    if left_fit_line is not None and right_fit_line is not None:
        left_start = [left_fit_line[0], left_fit_line[1]]
        left_end = [left_fit_line[2], left_fit_line[3]]
        right_start = [right_fit_line[0], right_fit_line[1]]
        right_end = [right_fit_line[2], right_fit_line[3]]
        intersection = line_intersection((left_start, left_end), (right_start, right_end))

        car_mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(car_mask, [np.array([(intersection[0], 50), left_start, right_start], np.int32)], match_mask_color)
        car_masked_image = cv2.bitwise_and(image, car_mask)
        car_roi_gray = cv2.cvtColor(car_masked_image, cv2.COLOR_RGB2GRAY)
        cars = car_cascade.detectMultiScale(car_roi_gray, 1.4, 1, minSize=(80, 80))

        for (x, y, w, h) in cars:
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        center = offset(left_fit_line[0], 180, right_fit_line[0])
        if abs(center) > 1.5:
            center_x = int(640 / 2.0)
            center_y = int(360 / 2.0)
            thickness = 2
            location = (center_x - 200, center_y - 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 3.5
            cv2.putText(temp_image, 'Warning', location, font, fontScale, (0, 0, 255), thickness)
            line_color = [0, 0, 255]
            lane_departure_detected = True
        else:
            lane_departure_detected = False

    if left_fit_line is not None:
        draw_fit_line(temp_image, left_fit_line, line_color)

    if right_fit_line is not None:
        draw_fit_line(temp_image, right_fit_line, line_color)

    image_with_lines = cv2.addWeighted(temp_image, 0.8, image, 1, 0.0)
    return image_with_lines, center, lane_departure_detected
