import cv2

def main():
    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    # 웹캠이 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 웹캠에서 프레임 읽기 및 표시
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        cv2.imshow('Webcam', frame)  # 프레임 표시

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 작업 완료 후 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
