import cv2 as cv
import os
import shutil

for img_fname in os.listdir('pngs'):
    img = cv.imread('pngs/' + img_fname)

    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    cv.imshow(img_fname, img)

    key = cv.waitKey(0)
    if key == ord('y'):
        shutil.copyfile('pngs/' + img_fname, 'RCTG-png/' + img_fname)
        cv.destroyAllWindows()
    elif key == ord('n'):
        cv.destroyAllWindows()
    elif key == 27:
        cv.destroyAllWindows()
        break
