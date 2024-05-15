import cv2
import time

# 영상 검출기
def faceDetector(cam, cascade):
    while True:
        start_t = time.time()
        
        # 캡처 이미지 불러오기
        ret, img = cam.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(20, 20))
        
        for (x, y, w, h) in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # FPS 계산 및 화면에 표시
        end_t = time.time()
        fps = 1.0 / (end_t - start_t)
        cv2.putText(img, f'FPS: {int(fps)}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 영상 출력
        cv2.imshow('facenet', img)
        
        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cam.release()
    cv2.destroyAllWindows()

# 가중치 파일 경로
cascade_filename = './sample/haarcascade_frontalface_alt.xml'

# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# 웹캠 열기
cam = cv2.VideoCapture(0)

# 영상 탐지기
faceDetector(cam, cascade)
