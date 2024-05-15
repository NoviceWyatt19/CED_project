import cv2

# 캠 실행
capture = cv2.VideoCapture(0)

# haarcascade 호출
face_cascade = cv2.CascadeClassifier('./sample/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./sample/haarcascade_eye.xml')

# 무한 루프 
while True:
    # frame: 영상을 받아 저장하는 변수, ret: 영상을 받고 있음을 체크할 boolean 변수
    ret, frame = capture.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 영상을 흑백으로 저장 -> 처리부담을 줄여 정확성 상승 기대
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 얼굴에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 얼굴 내부에서 눈 찾기
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2, cv2.LINE_4)

    # 영상 아래에 "eyes"라 표시하여 다른 코드와 구별할 수 있도록 함
    cv2.imshow("eyes", frame)
    
    # q를 누르면 함수 종료
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
