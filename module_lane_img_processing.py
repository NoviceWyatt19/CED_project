import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_region_of_interest(img, vertices):
    cv2.polylines(img, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

def get_fitline(img, lines):
    try:
        lines = np.squeeze(lines)
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
