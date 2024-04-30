import serial
import threading
import time

class SerialComm:
    def __init__(self):
        self.comm=serial.Serial("/dev/ttyAMA0", 9600)
        self.isWorking=True

        #self.inimsg = ser.readline().decode().strip()
        #print(self.inimsg)

    def writer(self, alcoltrue):
        if self.comm and self.comm.is_open:
            msg_alcol=str(alcoltrue)+'*'
            self.comm.write(msg_alcol.encode())

###############################################

def main():
    sc1=SerialComm()
    #sc1=serial_comm()
    def serial_read():
        sc1.writer(1)
        if sc1.distance < 10:
            pass

    serial_reader=threading.Thread(target = serial_read)
    serial_reader.start()
    
if __name__ == "__main__":
    main()