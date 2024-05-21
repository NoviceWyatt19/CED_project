import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

# 눈감김 비율 계산---------------------------------------
video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./sam/shape_predictor_68_face_landmarks.dat")

right_eye_points = list(range(36, 42))
left_eye_points = list(range(42, 48))

count_ar = 0
left_ar = [0] * 30
right_ar = [0] * 30 

eye_ratio_limit = 0.00

count_time = 0
count_time2 = 0

eye_cap = False
eye_open_done = True
program_switch = False
message_popup = False
print_counter = 0
txt_switch = False
txt_switch2 = False
alarm = False

face_alarm = False
face_reco = False

face_reco = False
face_reco_n = True
face = 0
fnd_count = 0

open_eye = True

# 눈 비율 값 계산 define-------------------------------
def eye_ratio(eyepoint):
    A = dist.euclidean(eyepoint[1], eyepoint[5])
    B = dist.euclidean(eyepoint[2], eyepoint[4])
    C = dist.euclidean(eyepoint[0], eyepoint[3])
    ER = (A + B) / (2.0 * C)

    return ER

# 얼굴 회전시 점 회전 define-----------------------------------
def rotate(brx, bry):
    crx = brx - midx
    cry = bry - midy
    arx = np.cos(-angle) * crx - np.sin(-angle) * cry
    ary = np.sin(-angle) * crx + np.cos(-angle) * cry
    rx = int(arx + midx)
    ry = int(ary + midy)

    return (rx, ry)

# 동영상 좌우반전 후 gray화 clahe 후 face detector--------------
while True:
    ret, frame = video_capture.read()
    flip_frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    detection = face_detector(clahe_image)

    # 화면내 인터페이스 작성 (메세지 팝업)----------------------------
    key = cv2.waitKey(10) & 0xFF

    if message_popup == True:
        if print_counter == 0:
            cv2.putText(flip_frame, "", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 1:
            cv2.putText(flip_frame, "Try again", (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 2:
            cv2.putText(flip_frame, "Gaze the camera", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if print_counter == 3:
            cv2.putText(flip_frame, "Program starts in : 3", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 4:
            cv2.putText(flip_frame, "Program starts in : 2", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 5:
            cv2.putText(flip_frame, "Program starts in : 1", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if print_counter == 6:
            cv2.putText(flip_frame, "CALCULATING", (240, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 프로그램 시작 분기문--------------------------------------------
    if key == ord("p"):  # p키를 눌렀을시 측정시작
        if not eye_cap == True:
            eye_open_done = False
        else:
            eye_open_done = True
            eye_cap = False
            cv2.destroyWindow("image2")
            program_switch = False
            eye_ratio_limit = 0.00
            left_ar = [0, ]
            right_ar = [0, ]

        count_ar = 0
        #(replace)winsound.PlaySound(None, winsound.SND_ASYNC)
        txt_switch = False
        alarm = False

    # 다시 누르면 이미지창 제거 메세지창 제거
    if eye_open_done == False:
        pass  # 아직 눈 측정이 진행 중일 때

    else:
        # 측정 시작
        for fd in detection:
            eye_open_shape = shape_predictor(clahe_image, fd)
            eye_open_landmarks = np.matrix([[p.x, p.y] for p in eye_open_shape.parts()])
            eye_open_left_eye = eye_open_landmarks[left_eye_points]
            eye_open_right_eye = eye_open_landmarks[right_eye_points]
            eye_open_ER_left = eye_ratio(eye_open_left_eye)
            eye_open_ER_right = eye_ratio(eye_open_right_eye)

            # ER값 측정 시작
            if (count_ar < 100):
                count_ar += 1

            for i in range(36, 41):
                cv2.line(flip_frame, (eye_open_shape.part(i).x, eye_open_shape.part(i).y),
                         (eye_open_shape.part(i + 1).x, eye_open_shape.part(i + 1).y), (255, 0, 0), 1)
                cv2.line(flip_frame, (eye_open_shape.part(41).x, eye_open_shape.part(41).y),
                         (eye_open_shape.part(36).x, eye_open_shape.part(36).y), (255, 0, 0), 1)
            for i in range(42, 47):
                cv2.line(flip_frame, (eye_open_shape.part(i).x, eye_open_shape.part(i).y),
                         (eye_open_shape.part(i + 1).x, eye_open_shape.part(i + 1).y), (255, 0, 0), 1)
                cv2.line(flip_frame, (eye_open_shape.part(47).x, eye_open_shape.part(47).y),
                         (eye_open_shape.part(42).x, eye_open_shape.part(42).y), (255, 0, 0), 1)

            print_counter = 2
            message_popup = True

            if (30 < count_ar <= 60):
                left_ar.append(eye_open_ER_left)
                right_ar.append(eye_open_ER_right)
                print_counter = 6

            if (60 < count_ar <= 70):
                print_counter = 0
                Max_ER_left = max(left_ar)
                Max_ER_right = max(right_ar)
                eye_ratio_limit = (Max_ER_left + Max_ER_right) / 2 * 0.65

            if (70 < count_ar <= 80):
                print_counter = 3
            if (80 < count_ar <= 90):
                print_counter = 4
            if (90 < count_ar < 100):
                print_counter = 5

            # 얼굴이 인식되는 동안 count_ar이 올라가면서 어레이에 저장후 최대값으로 설정, 메시지 팝업
            if (count_ar == 100):
                eye_open_done = True
                eye_cap = True
                program_switch = True
                print_counter = 0
                count_ar = 0
                count_time = time.time()  # count_ar이 최대일떄 측정 중단, 프로그램 시작

    # 얼굴 인식 범위 지정 및 얼굴 재정렬-----------------------------
    # 프로그램 시작
    if program_switch == True:
        face_reco = False
        face_reco_n = True

    for d in detection:
        face_reco = True
        fnd_count = 0
        count_time2 = time.time()

        if txt_switch2 == True:  # 얼굴 인식 불가 알람이 ON일때 알람을 끔
            #(replace)winsound.PlaySound(None, winsound.SND_ASYNC)
            face_alarm = False
            txt_switch2 = False

        x = d.left()
        y = d.top()
        x1 = d.right()
        y1 = d.bottom()  # d 값 저장
        bdx = x - (x1 - x) / 2
        bdy = y - (y1 - y) / 2
        bdx1 = x1 + (x1 - x) / 2
        bdy1 = y1 + (y1 - y) / 2
        midx = (x + x1) / 2
        midy = (y + y1) / 2
        shape = shape_predictor(clahe_image, d)
        rex = shape.part(45).x
        rey = shape.part(45).y
        lex = shape.part(36).x
        ley = shape.part(36).y
        mex = int (lex + (rex-lex)/2)
        mey = int (ley + (rey-ley)/2)
        #눈의 양끝점 좌표 설정 및 눈 사이 가운데 점 설정
        tanx = mex - lex
        tany = ley - mey
        tan = tany/tanx
        #tan 값 계산
        angle = np.arctan(tan)
        degree = np.degrees(angle)
        #각도 계산
        rsd_1 = rotate(x,y)
        rsd_2 = rotate(x1,y)
        rsd_3 = rotate(x,y1)
        rsd_4 = rotate(x1,y1)
        d2_1 = rotate(bdx,bdy)
        d2_2 = rotate(bdx1,bdy)
        d2_3 = rotate(bdx,bdy1)
        d2_4 = rotate(bdx1,bdy1)
        #좌표 회전
        pts1 = np.float32([[d2_1[0],d2_1[1]],[d2_2[0],d2_2[1]],[d2_3[0],d2_3[1]],[d2_4[0],d2_4[1]]])
        pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(flip_frame,M,(400,400))
        #회전된 좌표를 이용하여 새로운 창으로 프린트


        cv2.line(flip_frame, (int(midx), int(bdy)), (int(midx), int(bdy1)), (255, 255, 255), 1)
        cv2.line(flip_frame, (int(bdx), int(midy)), (int(bdx1), int(midy)), (255, 255, 255), 1)

        fshape = shape_predictor(clahe_image, d)
        landmarks = np.matrix([[p.x, p.y] for p in fshape.parts()])
        left_eye = landmarks[left_eye_points]
        right_eye = landmarks[right_eye_points]

        ER_left = eye_ratio(left_eye)
        ER_right = eye_ratio(right_eye)

        # 실시간으로 눈인식 후 ER계산
        open_eye = True

        if (ER_left < eye_ratio_limit) and (ER_right < eye_ratio_limit):
            if count_time == 0:
                count_time = time.time()
            if (time.time() - count_time) > 2:  # 2초 이상 감을시 알람 및 메세지 팝업
                if txt_switch == False:  # 알람이 ON일때는 알람 무시
                    #(replace)winsound.PlaySound("./sam/eyes_alarm.wav", winsound.SND_ASYNC)
                    txt_switch = True
                    alarm = True
                cv2.putText(flip_frame, "WAKE UP", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            open_eye = False

        else:
            count_time = 0
            txt_switch = False
            alarm = False

    if face_reco == False:
        if face_reco_n == True:
            count_time2 = time.time()
            face_reco_n = False

        if (time.time() - count_time2) > 5:
            if txt_switch2 == False:  # 알람이 ON일때는 알람 무시
                #(replace)winsound.PlaySound("./sam/eyes_alarm.wav", winsound.SND_ASYNC)
                face_alarm = True
                txt_switch2 = True
                fnd_count += 1

            if fnd_count > 2:
                eye_open_done = False
                eye_cap = False
                cv2.destroyWindow("image2")
                program_switch = False
                eye_ratio_limit = 0.00
                left_ar = [0, ]
                right_ar = [0, ]

                #(replace)winsound.PlaySound(None, winsound.SND_ASYNC)
                txt_switch = False
                alarm = False
                count_ar = 0
                count_time = 0
                count_time2 = 0
                face_alarm = False
                face_reco = False
                face_reco_n = True
                face = 0
                fnd_count = 0

    cv2.imshow("Frame", flip_frame)

    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
