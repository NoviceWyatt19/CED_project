import serial
import threading
import time

ser = serial.Serial('/dev/ttyACMO', 9600)

while True:
    #read data friom arduino
    data = ser.readline().decode().strip()
    print("Received: ", data)

    