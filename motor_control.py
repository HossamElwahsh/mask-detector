import RPi.GPIO as GPIO
from time import sleep

# Pins for Motor Driver Inputs 
Motor1A = 23
Motor1B = 24
Motor1E = 25


def setup():
    GPIO.setmode(GPIO.BCM)  # GPIO Numbering
    GPIO.setup(Motor1A, GPIO.OUT)  # All pins as Outputs
    GPIO.setup(Motor1B, GPIO.OUT)
    GPIO.setup(Motor1E, GPIO.OUT)


def loop():
    print('going forward\n')
    # Going forwards
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor1E, GPIO.HIGH)

    print('waiting 3 seconds\n')
    sleep(2)
    print('pausing\n')
    # Pause
    GPIO.output(Motor1E, GPIO.LOW)
    print('waiting 2 seconds\n')
    sleep(1)
    GPIO.output(Motor1E, GPIO.HIGH)
    print('going backward\n')

    # Going backwards
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor1E, GPIO.HIGH)

    print('waiting 3 seconds\n')
    sleep(2)
    # Stop
    GPIO.output(Motor1E, GPIO.LOW)
    print('waiting 1 minute\n')
    sleep(1)
    loop()

def destroy():
    GPIO.cleanup()


if __name__ == '__main__':  # Program start from here
    setup()
    try:
        loop()
    except KeyboardInterrupt:
        destroy()
    finally:
       print("clean up")
       GPIO.cleanup() # cleanup all GPIO