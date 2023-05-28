# TODO fix save path for screenshot

#-------------------FINAL FUNCTIONS---------------------

def by_image(self, template, accuracy=0.95, second_try=True):
    '''
    template - path to template image of control.\n
    accuracy - the percentage of pixels matching the template image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of template image has been changed. Default False.\n
    similarity - an integer on a five-point scale, sets the minimum similarity of the template image and the one found on the screen.
    Used with second_try=True. Default 4.\n
    Second search method works better in cases when the size of the template image has been increased.
    '''
    import cv2 as cv
    import numpy as np
    import imutils

    # import pywinauto
    # img = self.capture_as_image()
    # img_array = np.array(img.convert('RGB'))
    # img_cv = img_array[:, :, ::-1].copy()  # -1 does RGB -> BGR
    # screenshot = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)

    # DEBUG
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    h, w = template.shape

    # decrease
    found_dec = (0, 0, 0, 0)
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])

        if resized.shape[0] < h or resized.shape[1] < w:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_dec[0]:
            found_dec = (maxVal, maxLoc, r, result)
        else:
            break
    # increase
    found_inc = (0, 0, 0, 0)
    for scale in np.linspace(0.2, 1.0, 20):
        resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * (1+scale)))
        r = screenshot.shape[1] / float(resized.shape[1])

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_inc[0]:
            found_inc = (maxVal, maxLoc, r, result)
        else:
            break

    if found_dec[0] > found_inc[0]:
        (maxVal, maxLoc, r, res) = found_dec
    else:
        (maxVal, maxLoc, r, res) = found_inc

    if maxVal >= accuracy:
        # DEBUG
        cv.rectangle(screenshot, maxLoc,
                     (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
        x, y = int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2
        # control_wrapper = pywinauto.Desktop(backend=self.backend.name).from_point(x, y)
        # if 'top_level_only' not in control_wrapper:
        #         control_wrapper['top_level_only'] = False
        # control_wrapper['backend'] = self.backend.name
        # control_wrapper['parent'] = self.element_info

        # return pywinauto.WindowSpecification(control_wrapper)
    elif second_try and maxLoc[1]-1>=0 and maxLoc[0]-1>=0 and maxLoc[1]+1<res.shape[0] and maxLoc[0]+1<res.shape[1]:
        # коэффициент 0.02 получен экспериментально
        similarity = accuracy * 0.02
        val00 = res[maxLoc[1]-1][maxLoc[0]-1]
        val10 = res[maxLoc[1]][maxLoc[0]-1]
        val20 = res[maxLoc[1]+1][maxLoc[0]-1]
        val01 = res[maxLoc[1]-1][maxLoc[0]]
        val21 = res[maxLoc[1]+1][maxLoc[0]]
        val02 = res[maxLoc[1]-1][maxLoc[0]+1]
        val12 = res[maxLoc[1]][maxLoc[0]+1]
        val22 = res[maxLoc[1]+1][maxLoc[0]+1]
        # Make sure that vertical and horizontal values way bigger than in the corners
        valid_score = 0
        if val10 - (val00+val20)/2 >= similarity:
            valid_score += 1
        if val01 - (val00+val02)/2 >= similarity:
            valid_score += 1
        if val21 - (val20+val22)/2 >= similarity:
            valid_score += 1
        if val12 - (val02+val22)/2 >= similarity:
            valid_score += 1
        
        if valid_score >= 2:
            x, y = int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2

            # DEBUG
            print(x, y, max_value)
            with open("values.txt", 'w') as f:
                f.write(
                    f'{val00:.2f} {val01:.2f} {val02:.2f}\n{val10:.2f} {max_value:.2f} {val12:.2f}\n{val20:.2f} {val21:.2f} {val22:.2f}\n')
            cv.rectangle(screenshot, max_location,
                         (max_location[0] + w, max_location[1] + h), (0, 0, 255), 2)

            # control_wrapper = pywinauto.Desktop(backend=self.backend.name).from_point(x, y)

            # if 'top_level_only' not in control_wrapper:
            #         control_wrapper['top_level_only'] = False
            # control_wrapper['backend'] = self.backend.name
            # control_wrapper['parent'] = self.element_info

            # return pywinauto.WindowSpecification(control_wrapper)
        else:
            print('cannot find second try')
    else:
        print('cannot find first try')

    # DEBUG
    from matplotlib import pyplot as plt
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(screenshot)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()

def locate_one(template, accuracy=0.95, second_try=True):
    import cv2 as cv
    import numpy as np
    import imutils

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    h, w = template.shape
    box = (-1, -1, -1, -1)

    found_dec = (0, 0, 0, 0)
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])

        if resized.shape[0] < h or resized.shape[1] < w:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_dec[0]:
            found_dec = (maxVal, maxLoc, r, result)
        else:
            break

    found_inc = (0, 0, 0, 0)
    for scale in np.linspace(0.04, 0.8, 19):
        resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * (1+scale)))
        r = screenshot.shape[1] / float(resized.shape[1])

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_inc[0]:
            found_inc = (maxVal, maxLoc, r, result)
        else:
            break

    if found_dec[0] > found_inc[0]:
        (maxVal, maxLoc, r, res) = found_dec
    else:
        (maxVal, maxLoc, r, res) = found_inc

    if maxLoc == 0:
        return box

    if maxVal >= accuracy:
       box = (int(maxLoc[0] * r), int(maxLoc[1] * r), int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2)
    elif second_try and 0 < maxLoc[1] < res.shape[0]-1 and 0 < maxLoc[0] < res.shape[1]-1:
        similarity = accuracy * 0.02
        val00 = res[maxLoc[1]-1][maxLoc[0]-1]
        val10 = res[maxLoc[1]][maxLoc[0]-1]
        val20 = res[maxLoc[1]+1][maxLoc[0]-1]
        val01 = res[maxLoc[1]-1][maxLoc[0]]
        val21 = res[maxLoc[1]+1][maxLoc[0]]
        val02 = res[maxLoc[1]-1][maxLoc[0]+1]
        val12 = res[maxLoc[1]][maxLoc[0]+1]
        val22 = res[maxLoc[1]+1][maxLoc[0]+1]

        valid_score = 0
        if val10 - (val00+val20)/2 >= similarity:
            valid_score += 1
        if val01 - (val00+val02)/2 >= similarity:
            valid_score += 1
        if val21 - (val20+val22)/2 >= similarity:
            valid_score += 1
        if val12 - (val02+val22)/2 >= similarity:
            valid_score += 1
        if valid_score >= 2:
           box = (int(maxLoc[0] * r), int(maxLoc[1] * r), int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2)

    return box

def locate_all(template, count, accuracy=0.95, second_try=True):
    import cv2 as cv
    import numpy as np
    import imutils

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    h, w = template.shape
    height, width = screenshot.shape
    boxes=[]

    for j in range(count):
        found_dec = (0, 0, 0, 0)
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * scale))
            r = screenshot.shape[1] / float(resized.shape[1])

            if resized.shape[0] < h or resized.shape[1] < w:
                break

            result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

            if maxVal > found_dec[0]:
                found_dec = (maxVal, maxLoc, r, result)
            else:
                break

        found_inc = (0, 0, 0, 0)
        for scale in np.linspace(0.04, 0.8, 19):
            resized = imutils.resize(screenshot, width=int(screenshot.shape[1] * (1+scale)))
            r = screenshot.shape[1] / float(resized.shape[1])

            result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

            if maxVal > found_inc[0]:
                found_inc = (maxVal, maxLoc, r, result)
            else:
                break

        if found_dec[0] > found_inc[0]:
            (maxVal, maxLoc, r, res) = found_dec
        else:
            (maxVal, maxLoc, r, res) = found_inc
        if maxLoc == 0:
            for i in range(j, count):
                boxes.append((-1, -1, -1, -1))
            return boxes

        if maxVal >= accuracy:
            boxes.append(int(maxLoc[0] * r), int(maxLoc[1] * r), int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2)
        elif second_try and maxLoc[1]-1>=0 and maxLoc[0]-1>=0 and maxLoc[1]+1<res.shape[0] and maxLoc[0]+1<res.shape[1]:
            # коэффициент 0.02 получен экспериментально
            similarity = accuracy * 0.02
            val00 = res[maxLoc[1]-1][maxLoc[0]-1]
            val10 = res[maxLoc[1]][maxLoc[0]-1]
            val20 = res[maxLoc[1]+1][maxLoc[0]-1]
            val01 = res[maxLoc[1]-1][maxLoc[0]]
            val21 = res[maxLoc[1]+1][maxLoc[0]]
            val02 = res[maxLoc[1]-1][maxLoc[0]+1]
            val12 = res[maxLoc[1]][maxLoc[0]+1]
            val22 = res[maxLoc[1]+1][maxLoc[0]+1]
            # Make sure that vertical and horizontal values way bigger than in the corners
            valid_score = 0
            if val10 - (val00+val20)/2 >= similarity:
                valid_score += 1
            if val01 - (val00+val02)/2 >= similarity:
                valid_score += 1
            if val21 - (val20+val22)/2 >= similarity:
                valid_score += 1
            if val12 - (val02+val22)/2 >= similarity:
                valid_score += 1
            if valid_score >= 2:
                boxes.append(int(maxLoc[0] * r), int(maxLoc[1] * r), int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2)
        else:
            for i in range(j, count):
                boxes.append((-1, -1, -1, -1))
            return boxes

        for i in range(0, h):
            for k in range(0, w):
                if int(maxLoc[1] * r)+i < height and int(maxLoc[0] * r)+k < width:
                    screenshot[int(maxLoc[1] * r)+i][int(maxLoc[0] * r)+k] = 0

    return boxes


#-------------------TEST FUNCTIONS----------------------

# MY TEMPLATE MATCHING

def my_locate_one(template, accuracy=0.95, second_try=False):
    '''
    template - path to template image of control.\n
    accuracy - the percentage of pixels matching the template image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of template image has been changed. Default False.\n
    Second search method works better in cases when the size of the template image has been increased.
    Function returns rectangle coordinates (upper_left_x, upper_left_y, lower_right_x, lower_right_y) of found control.
    Size of rectangle are equal to template image size.
    '''
    import cv2 as cv
    import numpy as np
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    # Apply template Matching
    res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)

    _, max_value, _, max_location = cv.minMaxLoc(res)
    h, w = template.shape
    box = (-1, -1, -1, -1)

    # DEBUG
    x = max_location[0] + w//2
    y = max_location[1] + h//2
    print(x, y, max_value)

    if max_value >= accuracy:
        box = (max_location[0], max_location[1],
               max_location[0] + w, max_location[1] + h)

        # DEBUG
        cv.rectangle(screenshot, max_location,
                     (max_location[0] + w, max_location[1] + h), (0, 0, 255), 2)

    elif second_try and max_location[1]-1>=0 and max_location[0]-1>=0 and max_location[1]+1<res.shape[0] and max_location[0]+1<res.shape[1]:
        # коэффициент 0.02 получен экспериментально
        similarity = accuracy * 0.02

        val00 = res[max_location[1]-1][max_location[0]-1]
        val10 = res[max_location[1]][max_location[0]-1]
        val20 = res[max_location[1]+1][max_location[0]-1]

        val01 = res[max_location[1]-1][max_location[0]]
        val21 = res[max_location[1]+1][max_location[0]]

        val02 = res[max_location[1]-1][max_location[0]+1]
        val12 = res[max_location[1]][max_location[0]+1]
        val22 = res[max_location[1]+1][max_location[0]+1]

        # Make sure that vertical and horizontal values way bigger than in the corners
        valid_score = 0
        if val10 - (val00+val20)/2 >= similarity:
            valid_score += 1
        if val01 - (val00+val02)/2 >= similarity:
            valid_score += 1
        if val21 - (val20+val22)/2 >= similarity:
            valid_score += 1
        if val12 - (val02+val22)/2 >= similarity:
            valid_score += 1

        if valid_score >= 2:
            box = (max_location[0], max_location[1],
                   max_location[0] + w, max_location[1] + h)

            # DEBUG
            cv.rectangle(screenshot, max_location,
                         (max_location[0] + w, max_location[1] + h), (0, 0, 255), 2)

    # DEBUG
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return box

def my_locate_all(template, count, accuracy=0.95, second_try=False):
    '''
    template - path to template image of control.\n
    count - expected number of controls to be found.\n
    accuracy - the percentage of pixels matching the template image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of template image has been changed. Default False.\n
    Second search method works better in cases when the size of the template image has been increased.
    Function returns a list of rectangle coordinates (upper_left_x, upper_left_y, lower_right_x, lower_right_y) of found controls.
    Sizes of rectangles are equals to template image size.
    '''
    import cv2 as cv
    import numpy as np
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return

    if similarity <= 0:
        similarity = 1
    similarity = similarity/100-0.001

    height, width = screenshot.shape
    h, w = template.shape
    boxes = []
    for i in range(0, count):
        # Apply template Matching
        res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv.minMaxLoc(res)

        # DEBUG
        x = max_location[0] + w//2
        y = max_location[1] + h//2
        print(x, y, max_value)

        if max_value >= accuracy:
            boxes.append(
                (max_location[0], max_location[1], max_location[0] + w, max_location[1] + h))
            for i in range(0, h):
                    for j in range(0, w):
                        if 0 <= max_location[1]+i < height and 0 <= max_location[0]+j < width:
                            screenshot[max_location[1]+i][max_location[0]+j] = 0    
        elif second_try and max_location[1]-1>=0 and max_location[0]-1>=0 and max_location[1]+1<res.shape[0] and max_location[0]+1<res.shape[1]:
            # коэффициент 0.02 получен экспериментально
            similarity = accuracy * 0.02

            val00 = res[max_location[1]-1][max_location[0]-1]
            val10 = res[max_location[1]][max_location[0]-1]
            val20 = res[max_location[1]+1][max_location[0]-1]

            val01 = res[max_location[1]-1][max_location[0]]
            val21 = res[max_location[1]+1][max_location[0]]

            val02 = res[max_location[1]-1][max_location[0]+1]
            val12 = res[max_location[1]][max_location[0]+1]
            val22 = res[max_location[1]+1][max_location[0]+1]

            # Make sure that vertical and horizontal values way bigger than in the corners
            valid_score = 0
            if val10 - (val00+val20)/2 >= similarity:
                valid_score += 1
            if val01 - (val00+val02)/2 >= similarity:
                valid_score += 1
            if val21 - (val20+val22)/2 >= similarity:
                valid_score += 1
            if val12 - (val02+val22)/2 >= similarity:
                valid_score += 1

            if valid_score >= 2:
                boxes.append(
                    (max_location[0], max_location[1], max_location[0] + w, max_location[1] + h))

                for i in range(0, h):
                    for j in range(0, w):
                        if 0 <= max_location[1]+i < height and 0 <= max_location[0]+j < width:
                            screenshot[max_location[1]+i][max_location[0]+j] = 0

    # DEBUG
    for box in boxes:
        cv.rectangle(screenshot, (box[0]+5, box[1]+5),
                     (box[2]-5, box[3]-5), (0, 0, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return boxes

# MULTI SCALE TEMPLATE MATCHING


def scale_locate_one(template, accuracy=0.95):
    # import the necessary packages
    import numpy as np
    import imutils
    import cv2 as cv
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    h, w = template.shape
    found_dec = (0, 0, 0)
    prev_max= [0, 0]
    next_min= [0, 0]
    # decrease
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < h or resized.shape[1] < w:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # DEBUG
        print(maxVal)
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int(maxLoc[0] * r + w), int(maxLoc[1]  * r +h))

        debug_res = resized.copy()
        cv.rectangle(debug_res, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.imshow("Result", debug_res)
        cv.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if maxVal > found_dec[0]:
            prev_max[0] = found_dec[0]
            found_dec = (maxVal, maxLoc, r)
        else:
            next_min[0]=maxVal
            break

    # DEBUG
    print()
    # increase
    found_inc = (0, 0, 0)
    for scale in np.linspace(0.04, 0.8, 19):
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * (1+scale)))
        r = screenshot.shape[1] / float(resized.shape[1])

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        # DEBUG
        print(maxVal)
        (startX, startY) = (maxLoc[0], maxLoc[1])
        (endX, endY) = (maxLoc[0] + w, maxLoc[1] +h)

        debug_res = resized.copy()
        cv.rectangle(debug_res, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv.imshow("Result", debug_res)
        cv.waitKey(0)

        if maxVal > found_inc[0]:
            prev_max[1]=found_inc[0]
            found_inc = (maxVal, maxLoc, r)
        else:
            next_min[1]=maxVal
            break

    i = 0
    if found_dec[0] > found_inc[0]:
        (maxVal, maxLoc, r) = found_dec
    else:
        i=1
        (maxVal, maxLoc, r) = found_inc

    if next_min[i]==0:
        next_min[i]=prev_max[i]
    # коэффициент 0.01 получен экспериментально
    if maxVal - (prev_max[i]+next_min[i])/2 < 0.01 * accuracy:
        return (-1, -1, -1, -1)
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int(maxLoc[0] * r + w), int(maxLoc[1]  * r +h))

    # DEBUG
    print()
    cv.rectangle(screenshot, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return (startX, startY, endX, endY)

# KEYPOINT MATCHING


def keypoint_locate_one(template, accuracy=0.95):
    import numpy as np
    import cv2 as cv
    from PIL import ImageGrab
    import math

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    w, h = template.shape[::-1]
    box = (-1, -1, -1, -1)

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    if des1 is None or des2 is None:
        return box

    # BFMatcher
    best_matches = []

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # DEBUG
    good = []
    '''
    Short version: each keypoint of the first image is matched with a number of keypoints from the second image.
    We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement).
    Lowe's test checks that the two distances are sufficiently different.
    If they are not, then the keypoint is eliminated and will not be used for further calculations.
    https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html#:~:text=To%20filter%20the%20matches%2C%20Lowe,value%20is%20below%20a%20threshold.
    '''
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < (0.7 + (1-accuracy)*0.3)*n.distance:
            best_matches.append(m)
            # DEBUG
            good.append([m])
    
    if len(best_matches)==0:
        return box

    # DEBUG
    # cv.drawMatchesKnn expects list of lists as matches.
    import matplotlib.pyplot as plt
    img3 = cv.drawMatchesKnn(template, kp1, screenshot, kp2, good,
                             None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

    # Initialize lists
    list_x = []
    list_y = []

    # For each match...
    for mat in best_matches:
        # Get the matching keypoints for screenshot
        screenshot_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x, y) = kp2[screenshot_idx].pt

        # Append to each list
        list_x.append(x)
        list_y.append(y)

    if len(list_x) < 1 or len(list_y) < 1:
        return box
    elif len(list_x) > 9 and len(list_y) > 9:
        max_location = np.mean(sorted(list_x)[math.ceil(0.1 * len(list_x)): math.ceil(-0.1 * len(list_x))]), np.mean(sorted(list_y)[math.ceil(0.1 * len(list_y)): math.ceil(-0.1 * len(list_y))])
    else:
        max_location = np.mean(list_x), np.mean(list_y)

    box = (int(max_location[0] - w//2), int(max_location[1] - h//2),
               int(max_location[0] + w//2), int(max_location[1] + h//2))

    # DEBUG
    cv.rectangle(screenshot, (box[0], box[1]),
                    (box[2], box[3]), (0, 0, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return box

def keypoint_locate_all(screenshot, template, count, accuracy=0.95):
    import numpy as np
    import cv2 as cv
    from sklearn.cluster import KMeans

    screenshot = cv.imread(screenshot, cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    w, h = template.shape[::-1]

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    # BFMatcher
    best_matches = []

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=count+1)
    '''
    Short version: each keypoint of the first image is matched with a number of keypoints from the second image.
    We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement).
    Lowe's test checks that the two distances are sufficiently different.
    If they are not, then the keypoint is eliminated and will not be used for further calculations.
    https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html#:~:text=To%20filter%20the%20matches%2C%20Lowe,value%20is%20below%20a%20threshold.
    '''
   
    for points in matches:
        if len(points)<count+1:
            continue
        for i in range(count):
            if points[i].distance < (0.7 + (1-accuracy)*0.3)*points[count].distance:
                best_matches.append(points[i])

    # Initialize list
    points=[]

    # For each match...
    for mat in best_matches:
        # Get the matching keypoints for screenshot
        screenshot_idx = mat.trainIdx

        # i - columns
        # j - rows
        # Get the coordinates
        (i, j) = kp2[screenshot_idx].pt

        points.append([i, j])

    clusters = KMeans(n_clusters=count, n_init="auto").fit(points)

    positions = clusters.cluster_centers_

    for pos in positions:
        box = (int(pos[0] - w//2), int(pos[1] - h//2),
               int(pos[0] + w//2), int(pos[1] + h//2))
        cv.rectangle(screenshot, (box[0], box[1]),
                    (box[2], box[3]), (0, 0, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return positions


# ADVANCED TEMPLATE MATCHING



def main():
    # by_image(None, 'template.png', second_try=True)
    # result = my_locate_all('template.png', 5, second_try=True)
    # result = my_locate_one('template.png', second_try=False)
    result = scale_locate_one('template.png')
    # result = keypoint_locate_one('template.png')
    # result = keypoint_locate_all('test2.png','template.PNG', 5)
    print(result)


if __name__ == "__main__":
    main()
