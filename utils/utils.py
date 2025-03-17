import cv2

def wait_for_exit():
    while cv2.waitKey(1) != ord("q"):
        pass
    cv2.destroyAllWindows()

def show_images(imgs, size=(500, 500)):
    for i in imgs:
        cv2.imshow(i[0], cv2.resize(i[1], size))

    wait_for_exit()
