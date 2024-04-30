import serial
import threading
import time

class SerialComm:
    def init(self):
        self.comm=serial.Serial("/dev/ttyAMA0", 9600)
        self.isWorking=True

        inimsg = ser.readline().decode().strip()
        print(inimsg)

    def write(self, alcoltrue):
        if self.comm is None:
            return 0
        msg_alcol=str(alcoltrue)+'*'
        self.comm.write(msg_alcol.encode())

###############################################

def main():
    sc1=serial_comm()
    def serial_read():
        sc1.write(1)
        if sc1.distance<10:
            pass

    serial_reader=threading.Thread(target = serial_read)
    serial_reader.start()