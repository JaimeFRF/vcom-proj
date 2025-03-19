import cv2
import math
import random

#Colors
color_red = (0,0,255)
color_green = (0,255,0)
color_blue = (255,0,0)
color_orange = (0,165,255)
color_yellow = (51,255,255)

def wait_for_exit():
    while cv2.waitKey(1) != ord("q"):
        pass
    cv2.destroyAllWindows()

def show_images(imgs_data, size=(500, 500)):
    for i in imgs_data:
        cv2.imshow(i["name"], cv2.resize(i["image"], size))

    wait_for_exit()

def draw_hough_lines(data, color=color_red, withText=False, textSize=1.5):
    lines = data["metadata"].get("lines", None) # lines data from previous function in pipeline
    if (lines is None):
        raise ValueError("Lines data must be defined previously in pipeline, in order to draw lines")
    
    img = data["orig_img"].copy()

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
        pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))

        # Draw the line
        img = cv2.line(img, pt1, pt2, color, 3)

        #draw text to describe the line properties
        if (withText):
            pt1_text = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2_text = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            text_pos = (int((pt1_text[0] + pt2_text[0]) /2), int((pt1_text[1] + pt2_text[1]) /2) + random.randint(5,50))
            cv2.putText(img, f"r:{rho}, ang:{theta}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, textSize, color, 1, cv2.LINE_AA)
        
    data["image"] = img
    return data