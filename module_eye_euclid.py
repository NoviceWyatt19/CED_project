import numpy as np

def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
