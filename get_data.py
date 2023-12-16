import cv2 as cv

test_image = 'RCTG/pngs/a/a_1.png'

img = cv.imread(test_image)

img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)

cv.imshow('test_image', img)
cv.waitKey(0)
cv.destroyAllWindows()
