import cv2
import numpy as np
from keras.models import load_model
from copy import copy, deepcopy
import time
import winsound

model = load_model("/Birinder/Programs/Python/Edge AI/Project2/mnist.h5")
ix, iy, k = 200, 200, 1


def onMouse(event, x, y, flag, param):
    global ix, iy, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        k = -1

w_max = 380
h_max = 200
w_min = 200
h_min = 200

def resize_by_height(image, height):

    h, w = image.shape[:2]

    ratio = h/w

    new_width = int(height/ratio)

    ret = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)

    if new_width > w_max:
        return resize_by_width(image, w_max)
    
    if new_width < w_min:
        pad = (w_min - new_width) //2
        ret = cv2.copyMakeBorder(ret, 0,0,pad,pad, borderType=cv2.BORDER_CONSTANT, value = (255,255,255))


    return ret

def resize_by_width(image, width):

    h, w = image.shape[:2]

    ratio = h/w

    new_height = int(ratio*width)

    ret = cv2.resize(image, (width, new_height))

    cv2.imshow("horizontal max", ret)

    if new_height < h_min:
        pad = (h_min - new_height) //2
        ret = cv2.copyMakeBorder(ret,pad,pad,0,0, borderType=cv2.BORDER_CONSTANT, value = (255,255,255))

    return ret



h_extra_max = 200

def add_overlays(to_add, height = h_extra_max, position = "topleft", placer = None):

    final_data_image = combined

    if len(to_add):

        final_data_image = cv2.copyMakeBorder(combined, h_extra_max,0,0,0, borderType=cv2.BORDER_CONSTANT, value = (255,255,255))
        cv2.imshow("test", final_data_image)
        # cv2.waitKey(0)
        print(len(to_add))

        if position == "topleft":
            y_offset, x_offset = 0, 0
        
        if position == "topright":
            y_offset = 0
            x_offset = final_data_image.shape[1] - placer.shape[1]

        l_w_added = 0

        for i, elem in enumerate(to_add):
            print(i, " is being added")
            print(len((elem, height)))
            resized = resize_by_height(elem, height)
            if len(resized.shape) < 3:
                resized = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
            h, w = resized.shape[:2]

            if h < height:
                final_data_image[height-h:h, l_w_added:l_w_added+w]

            final_data_image[0:h, l_w_added:l_w_added+w] = resized

            l_w_added+=w


    ########### Tracker ############

    if show_curr_track == True:
            # X, Y
        try:

            curr_track_img = frame2[max(0, curr_pts[1] - curr_track_vis):min(frame2.shape[0], curr_pts[1] + curr_track_vis), 
            max(0, curr_pts[0]-curr_track_vis): min(frame2.shape[1], curr_pts[0]+curr_track_vis)]

            # cv2.rectangle(combined, )

            cv2.imshow("tracker", curr_track_img)
            r_bound = final_data_image.shape[1]
            remaining_right = min(max_track_size, r_bound-l_w_added)
            resized = resize_by_width(curr_track_img, remaining_right)
            h, w = resized.shape[:2]
            final_data_image[0:h, r_bound-w:r_bound] = resized
            # combined[50:50+curr_track_total, 200:200+curr_track_total] = curr_track_img
            cv2.imshow("combined check", combined)
            # cv2.waitKey(0)

        except:
            pass

    return final_data_image


last_centre = None

cv2.namedWindow("window")
cv2.setMouseCallback("window", onMouse)

# cap = cv2.VideoCapture("/Birinder/Programs/Python/Edge AI/Project2/vid.mp4")
i = cv2.imread("/Birinder/Programs/Python/Edge AI/Project2/img.png")

cap = cv2.VideoCapture(0)

framect = 0

while True:
    _, frm = cap.read()

    cv2.imshow("window", frm)
    cv2.imshow("horizontal resize test", resize_by_width(frm, 200))
    cv2.imshow("vertical resize test", resize_by_height(frm, 200))

    if cv2.waitKey(1) == 27 or k == -1:
        old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        break

old_pts = np.array([[ix, iy]], dtype="float32").reshape(-1, 1, 2)
mask = np.zeros_like(frm)

show_crop = True
show_cv = True
show_curr_track = True
curr_track_vis = 20
curr_track_total = curr_track_vis*2
max_track_size = 100


# seven = cv2.imread("/Birinder/Programs/Python/Edge AI/Project2/img7.png")


# gray = cv2.cvtColor(seven.copy(), cv2.COLOR_BGR2GRAY)
# ret, th = cv2.threshold(
#     gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# seven_r = cv2.resize(th, (28,28))
# cv2.imshow("seven28", seven_r)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# seven = seven_r.reshape(1, 28, 28, 1)
# seven = seven / 255.0
# pred = model.predict([seven_r])[0]
# final_pred = np.argmax(pred)
# print(final_pred)


while True:
    ctr = 0
    framect+=1
    _, frame2 = cap.read()

    new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    new_pts, status, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        new_gray,
        old_pts,
        None,
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08),
    )

    curr_pts = (int(new_pts.ravel()[0]), int(new_pts.ravel()[1]))

    cv2.circle(mask, curr_pts, 2, (0, 255, 0), 5)
    combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)
    if last_centre:
        cv2.line(mask, last_centre, curr_pts, (0, 255, 0), 10)
    last_centre = (int(new_pts.ravel()[0]), int(new_pts.ravel()[1]))

    image = mask.copy()
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contourfinder = th-255

    # cv2.imshow("gray", gray)
    # cv2.imshow("th", th)

    contours = cv2.findContours(
        contourfinder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for i, cnt in enumerate(contours):

        to_add_in_final = []


        h_extra, w_extra = 0, 0
        h_store, w_store = [], []

        # print(f"{framect}, {ctr}")
        ctr+=1
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(combined, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]
        cropped_digit_for_final = digit.copy()
        # cv2.imshow(f"original digit{i}", digit)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))
        # cv2.imshow(f"resized_digit{i}", resized_digit)
        # cv2.imshow("actual_resized_digit", actual_resized_digit)

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)),
                              "constant", constant_values=255)

        digit = padded_digit.reshape(1, 28, 28, 1)
        # cv2.imshow(f"padded_digit{i}", padded_digit)
        cv_digit_for_final = padded_digit.copy()
        actual_resized_digit = cv2.resize(padded_digit, (400, 400))
        # cv2.imshow(f"actual_resized_digit{i}", actual_resized_digit)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
        # print(final_pred)

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(combined, data, (x, y - 5), font, fontScale, color, thickness)

        # cv2.imshow(f"DataStats for digit {i}")

        final_data_image = combined.copy()

        if show_crop ==True:
            to_add_in_final.append(cropped_digit_for_final)

        if show_cv == True:
            to_add_in_final.append(cv_digit_for_final)

        # curr_pts
        
            # except:
            #     pass

        # if show_curr_track == True:
        #     curr_pts = (int(new_pts.ravel()[0]), int(new_pts.ravel()[1]))

        for i,elem in enumerate(to_add_in_final):
            cv2.imshow(f"{i}", elem)

        final_data_image = add_overlays(to_add_in_final)
        cv2.imshow("Final Image Data", final_data_image)


        # full_data_window = np.pad(combined )


    # cv2.imshow("Drawing", mask)
    # cv2.imshow("Initial", combined)
    # cv2.imshow("Digit", image)

    old_gray = new_gray.copy()
    old_pts = new_pts.copy()

    def save():
        t = time.localtime()
        timestamp = time.strftime("%b-%d-%Y_%H%M", t)
        filename = "Export_" + timestamp + str(time.time()) + ".jpg"
        cv2.imwrite("/Birinder/Programs/Python/Edge AI/Project2/Exports/"+f"{filename}", final_data_image)
        winsound.MessageBeep()

    # if cv2.waitKey(1) == 115:

    # if cv2.waitKey(1) == 100:
    #     show_crop = not show_crop
    
    # if cv2.waitKey(1) == 99:
    #     show_cv = not show_cv

    # if cv2.waitKey(1) == 116:
    #     show_curr_track = not show_curr_track

    d_exec = {27: "cap.release();cv2.destroyAllWindows();break;",100:"show_crop = not show_crop", 99:"show_cv = not show_cv", 116:"show_curr_track = not show_curr_track", 115:"save()"}

    inp = cv2.waitKey(1)
    if inp in d_exec:
        exec(d_exec[inp])
    inp = None

    # if cv2.waitKey(1) == 27:
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     break
