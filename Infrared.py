import time

import RPi.GPIO as GPIO


class Infrared(object):
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(12, GPIO.IN)
        GPIO.setup(21, GPIO.OUT)

    def light_up(self):
        GPIO.output(21, GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(21, GPIO.HIGH)
        time.sleep(0.5)

    def detct(self):
        if GPIO.input(12) == True:
            print("someone is closing ")
            self.light_up()
            return True
        else:
            print("nobody is comming ")
            return False

    def done(self):
        GPIO.cleanup()
