import cv2
import time
import serial
import threading
from module_lane_detection import process_lane_detection
from utils import load_cascade, open_video, open_serial

lane_departure_detected = False
arduino_response = ""
stop_thread = False

def read_from_arduino(ser):
    global arduino_response, stop_thread
    while not stop_thread:
        try:
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting).decode().strip()
                if response:
                    arduino_response = response
                    print(f"Arduino response: {response}")
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            break

def send_signal_to_arduino(ser):
    try:
        print("Sending SLEEP_TRUE to Arduino")
        ser.write("SLEEP_TRUE\n".encode())
        ser.flush()
    except serial.SerialException as e:
        print(f"Error writing to serial port: {e}")

def main():
    global lane_departure_detected, stop_thread
    try:
        ser = open_serial("/dev/cu.usbmodemF412FA6F49D82", 9600)
    except IOError as e:
        print(e)
        return

    time.sleep(5)  # 시리얼 통신 안정화 대기

    # 아두이노로부터 데이터를 읽는 스레드 시작
    read_thread = threading.Thread(target=read_from_arduino, args=(ser,), daemon=True)
    read_thread.start()

    try:
        car_cascade = load_cascade('./cars.xml')
    except IOError as e:
        print(e)
        return
    
    try:
        cap = open_video('./change.avi')
    except IOError as e:
        print(e)
        return

    frame_count = 0
    check_rate = 10
    sent_alert = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            lane_frame, center, lane_departure_detected = process_lane_detection(frame, car_cascade, frame_count, check_rate, lane_departure_detected)
            cv2.imshow("Lane Detection", lane_frame)

            if lane_departure_detected and not sent_alert:
                print("Lane Departure Detected: Send signal to arduino")
                threading.Thread(target=send_signal_to_arduino, args=(ser,)).start()
                sent_alert = True
            elif not lane_departure_detected:
                sent_alert = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame_count += 1
        else:
            break

    stop_thread = True  # 스레드 종료 플래그 설정
    read_thread.join()  # 스레드가 안전하게 종료되도록 대기

    cv2.destroyAllWindows()
    cap.release()
    ser.close()

if __name__ == "__main__":
    main()
