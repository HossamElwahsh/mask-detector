import RPi.GPIO as GPIO
import time

ready = 7
faceRead = 5    # yellow
maskOn = 7      # green
maskOff = 3     # red

# led = 18
# switch = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ready, GPIO.OUT)
GPIO.setup(faceRead, GPIO.OUT)
GPIO.setup(maskOn, GPIO.OUT)
GPIO.setup(maskOff, GPIO.OUT)

GPIO.output(faceRead, GPIO.LOW)
GPIO.output(maskOn, GPIO.LOW)
GPIO.output(maskOff, GPIO.LOW)
GPIO.cleanup()
# while(1):
#     GPIO.output(ready, GPIO.HIGH)
#     time.sleep(0.3)
#     GPIO.output(ready, GPIO.LOW)
#     time.sleep(0.3)
    # print('Switch status = ', GPIO.input(switch))

# GPIO.cleanup()