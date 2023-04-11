# TODO fix save path for screenshot

# FINAL FUNCTION
def by_image(self, image, accuracy=0.95, second_try=False, similarity=4):
    '''
    image - path to original image of control.\n
    accuracy - the percentage of pixels matching the original image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of original image has been changed. Default False.\n
    similarity - an integer on a five-point scale, sets the minimum similarity of the original image and the one found on the screen.
    Used with second_try=True. Default 4.\n
    Second search method works better in cases when the size of the original image has been increased.
    '''
    import cv2 as cv
    import numpy as np

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

    if isinstance(image, str):
        template = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('cannot read image')
    else:
        print('invalid format of image')

    # Apply template Matching
    res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)
    a, max_value, b, max_location = cv.minMaxLoc(res)

    w, h = template.shape[::-1]
    x = max_location[0] + w//2
    y = max_location[1] + h//2

    if max_value >= accuracy:
        # DEBUG
        cv.rectangle(screenshot, max_location,
                     (max_location[0] + w, max_location[1] + h), (0, 0, 255), 2)

        # control_wrapper = pywinauto.Desktop(backend=self.backend.name).from_point(x, y)
        # if 'top_level_only' not in control_wrapper:
        #         control_wrapper['top_level_only'] = False
        # control_wrapper['backend'] = self.backend.name
        # control_wrapper['parent'] = self.element_info

        # return pywinauto.WindowSpecification(control_wrapper)
    elif second_try:
        # Getting each value in format 0.00
        val00 = round(res[max_location[1]-1][max_location[0]-1], 2)
        val01 = round(res[max_location[1]][max_location[0]-1], 2)
        val02 = round(res[max_location[1]+1][max_location[0]-1], 2)

        val10 = round(res[max_location[1]-1][max_location[0]], 2)
        val12 = round(res[max_location[1]+1][max_location[0]], 2)

        val20 = round(res[max_location[1]-1][max_location[0]+1], 2)
        val21 = round(res[max_location[1]][max_location[0]+1], 2)
        val22 = round(res[max_location[1]+1][max_location[0]+1], 2)

        '''
        it looks like that
        val00   val01   val02
        val10 max_value val12
        val20   val21   val22

        '''
        # DEBUG
        print(x, y, max_value)
        with open("values.txt", 'w') as f:
            f.write(
                f'{val00:.2f} {val01:.2f} {val02:.2f}\n{val10:.2f} {max_value:.2f} {val12:.2f}\n{val20:.2f} {val21:.2f} {val22:.2f}\n')

        # Make sure that vertical and horizontal values way bigger than in the corners
        if similarity <= 0:
            similarity = 1
        similarity = similarity/100-0.001
        if abs(val01-val00)+abs(val01-val02) >= similarity and abs(val10-val00)+abs(val10-val20) >= similarity \
                and abs(val21-val20)+abs(val21-val22) >= similarity and abs(val12-val02)+abs(val12-val22) >= similarity:

            # DEBUG
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

# MY TEMPLATE MATCHING


def locate_one(image, accuracy=0.95, second_try=False, similarity=4):
    '''
    image - path to original image of control.\n
    accuracy - the percentage of pixels matching the original image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of original image has been changed. Default False.\n
    similarity - an integer on a five-point scale, sets the minimum similarity of the original image and the one found on the screen.
    Used with second_try=True. Default 4.\n
    Second search method works better in cases when the size of the original image has been increased.
    Function returns rectangle coordinates (upper_left_x, upper_left_y, lower_right_x, lower_right_y) of found control.
    Size of rectangle are equal to original image size.
    '''
    import cv2 as cv
    import numpy as np
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(image, str):
        template = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('cannot read image')
    else:
        print('invalid format of image')

    # Apply template Matching
    res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)

    if similarity <= 0:
        similarity = 1
    similarity = similarity/100-0.001

    a, max_value, b, max_location = cv.minMaxLoc(res)
    w, h = template.shape[::-1]
    box = (0, 0, 0, 0)

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

    elif second_try:
        # Getting each value in format 0.00
        val00 = round(res[max_location[1]-1][max_location[0]-1], 2)
        val01 = round(res[max_location[1]][max_location[0]-1], 2)
        val02 = round(res[max_location[1]+1][max_location[0]-1], 2)

        val10 = round(res[max_location[1]-1][max_location[0]], 2)
        val12 = round(res[max_location[1]+1][max_location[0]], 2)

        val20 = round(res[max_location[1]-1][max_location[0]+1], 2)
        val21 = round(res[max_location[1]][max_location[0]+1], 2)
        val22 = round(res[max_location[1]+1][max_location[0]+1], 2)

        # Make sure that vertical and horizontal values way bigger than in the corners
        if abs(val01-val00)+abs(val01-val02) >= similarity and abs(val10-val00)+abs(val10-val20) >= similarity \
                and abs(val21-val20)+abs(val21-val22) >= similarity and abs(val12-val02)+abs(val12-val22) >= similarity:

            box = (max_location[0], max_location[1],
                   max_location[0] + w, max_location[1] + h)

            # DEBUG
            cv.rectangle(screenshot, max_location,
                         (max_location[0] + w, max_location[1] + h), (0, 0, 255), 2)

    # DEBUG
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return box


def locate_all(image, count, accuracy=0.95, second_try=False, similarity=4):
    '''
    image - path to original image of control.\n
    count - expected number of controls to be found.\n
    accuracy - the percentage of pixels matching the original image and the one found on the screen. Default 0.95.\n
    second_try - indicates whether to use a different search method based on the similarity score.
    Useful when the size of original image has been changed. Default False.\n
    similarity - an integer on a five-point scale, sets the minimum similarity of the original image and the one found on the screen.
    Used with second_try=True. Default 4.\n
    Second search method works better in cases when the size of the original image has been increased.
    Function returns a list of rectangle coordinates (upper_left_x, upper_left_y, lower_right_x, lower_right_y) of found controls.
    Sizes of rectangles are equals to original image size.
    '''
    import cv2 as cv
    import numpy as np
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(image, str):
        template = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('cannot read image')
    else:
        print('invalid format of image')

    if similarity <= 0:
        similarity = 1
    similarity = similarity/100-0.001

    height, width = screenshot.shape
    w, h = template.shape[::-1]
    boxes = []
    for i in range(0, count):
        # Apply template Matching
        res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)
        a, max_value, b, max_location = cv.minMaxLoc(res)

        # DEBUG
        x = max_location[0] + w//2
        y = max_location[1] + h//2
        print(x, y, max_value)

        if max_value >= accuracy:
            boxes.append(
                (max_location[0], max_location[1], max_location[0] + w, max_location[1] + h))
        elif second_try:
            # Getting each value in format 0.00
            val00 = round(res[max_location[1]-1][max_location[0]-1], 2)
            val01 = round(res[max_location[1]][max_location[0]-1], 2)
            val02 = round(res[max_location[1]+1][max_location[0]-1], 2)

            val10 = round(res[max_location[1]-1][max_location[0]], 2)
            val12 = round(res[max_location[1]+1][max_location[0]], 2)

            val20 = round(res[max_location[1]-1][max_location[0]+1], 2)
            val21 = round(res[max_location[1]][max_location[0]+1], 2)
            val22 = round(res[max_location[1]+1][max_location[0]+1], 2)

            # Make sure that vertical and horizontal values way bigger than in the corners
            if abs(val01-val00)+abs(val01-val02) >= similarity and abs(val10-val00)+abs(val10-val20) >= similarity \
                    and abs(val21-val20)+abs(val21-val22) >= similarity and abs(val12-val02)+abs(val12-val22) >= similarity:

                boxes.append(
                    (max_location[0], max_location[1], max_location[0] + w, max_location[1] + h))

        for i in range(0, h):
            for j in range(0, w):
                if 0 <= max_location[1]+i < height and 0 <= max_location[0]+j < width:
                    screenshot[max_location[1]+i][max_location[0]+j] = 0

    # DEBUG
    for box in boxes:
        cv.rectangle(screenshot, (box[0]+5, box[1]+5),
                     (box[2]-5, box[3]-5), (255, 255, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return boxes

# MULTI SCALE TEMPLATE MATCHING


def scale_locate_one(template):
    # import the necessary packages
    import numpy as np
    import imutils
    import cv2 as cv
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    template = cv.imread(template)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    (tH, tW) = template.shape[:2]
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # DEBUG
    cv.rectangle(screenshot, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv.imshow("Result", screenshot)
    cv.waitKey(0)

    return ((maxVal),  (startX, startY, endX, endY))

# KEYPOINT MATCHING


def keypoint_locate_one(template, custom_accuracy=False, accuracy=0.95):
    import numpy as np
    import cv2 as cv
    from PIL import ImageGrab

    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)

    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('cannot read image')
    else:
        print('invalid format of image')

    w, h = template.shape[::-1]

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    # BFMatcher
    best_matches = []
    if not custom_accuracy:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        best_matches = sorted(matches, key=lambda x: x.distance)
    else:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < accuracy*n.distance:
                best_matches.append(m)
                good.append([m])

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

    if len(list_x) > 0 and len(list_y) > 0:
        max_location = (np.mean(list_x), np.mean(list_y))
        box = (max_location[0] - w/2, max_location[1] - h/2,
               max_location[0] + w/2, max_location[1] + h/2)

        # DEBUG
        cv.rectangle(screenshot, (box[0], box[1]),
                     (box[2], box[3]), (0, 0, 255), 2)
        cv.imshow("Result", screenshot)
        cv.waitKey(0)

        return box

    return (0, 0, 0, 0)


def main():
    # by_image(None, 'original.png', second_try=True)
    # result = locate_all('original.png', 10, second_try=True)
    # result = locate_one('original.png', second_try=True)
    # result = scale_locate_one('original.png')
    result = keypoint_locate_one('original.png')
    print(result)


if __name__ == "__main__":
    main()
