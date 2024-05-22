import serial
import time
import random
import threading

# 시리얼 포트 설정 (적절한 포트를 사용하세요)
ser = serial.Serial('/dev/cu.usbmodemF412FA6F49D82', 9600)  # 포트 및 통신 속도에 맞게 설정
time.sleep(2)  # 시리얼 포트 초기화 시간

def send_sleep_command():
    while True:
        # 랜덤하게 SLEEP 상태 설정
        sleep_command = random.choice(["SLEEP_TRUE", "SLEEP_FALSE"])
        print(f"Sending sleep command: {sleep_command}")
        ser.write(f"{sleep_command}\n".encode())
        time.sleep(5)  # 5초마다 SLEEP 명령 전송

def receive_data():
    while True:
        if ser.in_waiting > 0:
            received_data = ser.readline().decode().strip()
            print("Received data:", received_data)

            # 받은 데이터에 따라 적절한 반응 결정
            if received_data == "Alcol_DETECTED":
                print("Alcol detected!")
                ser.write("SENSOR_1_ACK\n".encode())  # 아두이노로 응답 전송
            elif received_data == "CO2_DETECTED":
                print("CO2 detected!")
                ser.write("SENSOR_2_ACK\n".encode())  # 아두이노로 응답 전송

# 스레드 생성
send_thread = threading.Thread(target=send_sleep_command)
receive_thread = threading.Thread(target=receive_data)

# 스레드 시작
send_thread.start()
receive_thread.start()

# 메인 스레드가 종료되지 않도록 대기
send_thread.join()
receive_thread.join()
