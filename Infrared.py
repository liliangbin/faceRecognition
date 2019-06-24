import time

import RPi.GPIO as GPIO
import requests


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


if __name__ == '__main__':
    # inf = Infrared()
    # while True:
    #     if inf.detct():
    #         print("find a people")
    #     else:
    #         print("no people ")

    data = {
        "key": "7b97e1bed4d6a452ab5bc68b9fc1e681",
        "id": 1,
        "status": "unlock"
    }
    # response = requests.get("http://kuailezhai.cn/mobile/", data=data)
    response = requests.get('http://kuailezhai.cn/mobile/?key=7b97e1bed4d6a452ab5bc68b9fc1e681&id=1&status=unlock')

    print(response.text)
