import threading
import subprocess

def run_eye_detect():
    subprocess.run(["python", "app_eye_detect.py"])

def run_lane_detect():
    subprocess.run(["python", "app_lane_detect.py"])

def main():
    # 첫 번째 스레드에서 app_eye_detect.py 실행
    eye_detect_thread = threading.Thread(target=run_eye_detect)
    eye_detect_thread.start()

    # 두 번째 스레드에서 app_lane_detect.py 실행
    lane_detect_thread = threading.Thread(target=run_lane_detect)
    lane_detect_thread.start()

    # 모든 스레드가 종료될 때까지 대기
    eye_detect_thread.join()
    lane_detect_thread.join()

if __name__ == "__main__":
    main()
