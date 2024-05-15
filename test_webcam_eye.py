import cv2

#캠 실행
capture = cv2.VideoCapture(0)

# haarcascade 호출
eye_cascade = cv2.CascadeClassifier('./sample/haarcascade_eye.xml')

# 무한 루프 
while True:
    # frame: 영상을 받아 저장하는 변수, ret: 영상을 받고 있음을 체크할 boolean변수
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #영상을 흑백으로 저장 -> 처리부담을 줄여 정확성 상승 기대

    #scaleFactor는 1에 가까울 수록 성능 향상, minNeighbors를 높일 수록 검출률과 오탐지율 상승
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors= 4, minSize=(10,10))

    #눈을 찾으면 박스로 표시
    if len(eyes):
        for x, y, w, h in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_4)
    # 영상아래에 eyes라 표시하여 다른 코드와 구별할 수 있도록함
    cv2.imshow("eyes", frame)
    # q를 누르면 함수 종료
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# 눈만 따로 찾으면 외부의 다른 것들도 눈으로 인식하는 문제가 발생 -> 얼굴탐지 기능과 합쳐 사용해 오탐지율 개선