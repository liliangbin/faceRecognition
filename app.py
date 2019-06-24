import requests

from Infrared import Infrared
from read_camera import Camera_reader

if __name__ == '__main__':
    camera = Camera_reader()
    infrared = Infrared()

    while True:
        if infrared.detct():  # INFO 2019/6/12 21:03   红外传感器发现有人进入
            infrared.light_up()  # INFO 2019/6/12 21:03   亮灯
            if camera.get_one_picture():
                # INFO 2019/6/12 21:00  去请求打开门锁的接口。
                data = {
                    "key": "7b97e1bed4d6a452ab5bc68b9fc1e681",
                    "id": 1,
                    "status": "unlock"
                }
                response = requests.get("http://kuailezhai.cn/update/", data=data)
                print(response.text)
                print("是管理员，允许开门")
                pass

    infrared.done()
    camera.camera_done()
