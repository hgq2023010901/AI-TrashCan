import cv2 as cv

capture = cv.VideoCapture(0)
cnt = 0
name = "capture/cap"
while True:
    ret, image = capture.read()
    if ret is True:
        image = cv.flip(image, 1)
        cv.imshow("frame", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('p'):
            cv.imwrite(name+str(cnt)+".jpg",image)
            cnt = cnt+1
    else:
        break

cv.destroyAllWindows()