import threading
import subprocess
import time

def run_eye_main():
    subprocess.run(["python", "eye_main.py"])
    
def run_lane_main():
    subprocess.run(["python", "lane_main.py"])

if __name__ == "__main__":
    # 먼저 eye_main.py를 실행
    eye_thread = threading.Thread(target=run_eye_main)
    eye_thread.start()

    # eye_main.py가 시작될 시간을 기다림
    time.sleep(2)

    # lane_main.py 실행
    lane_thread = threading.Thread(target=run_lane_main)
    lane_thread.start()

    # 두 스레드가 종료될 때까지 기다림
    eye_thread.join()
    lane_thread.join()
