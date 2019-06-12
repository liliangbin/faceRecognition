import cv2

cap = cv2.VideoCapture(0)
i = 0
ret, frame = cap.read()
cv2.imwrite('./' + str(i) + '.jpg', frame)

cap.release()
#cap.destroyAllWindows()
# 拍照获得一张图片的信息
