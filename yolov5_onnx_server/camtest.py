import cv2 as cv

capture = cv.VideoCapture(0)
while True:
    ret, image = capture.read()
    if ret is True:
        image = cv.flip(image, 1)
        cv.imshow("frame", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()