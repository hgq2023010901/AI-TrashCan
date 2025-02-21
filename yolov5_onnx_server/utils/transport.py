import serial as ser
import time
a = "test01"
se =ser.Serial("/dev/ttyTHS1",9600,timeout=1)
while True:
    se.write(a.encode())
    time.sleep(2)
    print("send ok")
    print(se.is_open)
