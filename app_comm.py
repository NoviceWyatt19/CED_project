import serial
import time

def i_hate_aruino(self, path='/dev/cu.usbmodemF412FA6F49D82', rate=9600):
    self = serial.Serial(path, rate)
    time.sleep(2)

def i_hate_communication(self, path='/dev/cu.usbmodemF412FA6F49D82', send=''):
    self.write(f"{send}\n".encode())
    print(f'path= {path}')
    print(f'send= {send}')
    time.sleep(2)

