
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
    screenshot.save(r'C:/Users/Никита/Desktop/screen.png')
    screenshot = cv.imread('screen.png', 0)

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
                     (max_location[0] + w, max_location[1] + h), (0, 0, 255), 3)

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
                         (max_location[0] + w, max_location[1] + h), (0, 0, 255), 3)

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
    screenshot.save(r'C:/Users/Никита/Desktop/screen.png')
    screenshot = cv.imread('screen.png', 0)

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

    boxes = []
    for i in range(0, count):
        a, max_value, b, max_location = cv.minMaxLoc(res)
        w, h = template.shape[::-1]

        # DEBUG
        x = max_location[0] + w//2
        y = max_location[1] + h//2
        print(x, y, max_value)

        if max_value >= accuracy:
            boxes.append(
                (max_location[0], max_location[1], max_location[0] + w, max_location[1] + h))

            # DEBUG
            cv.rectangle(screenshot, max_location,
                         (max_location[0] + w, max_location[1] + h), (0, 0, 255), 3)

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

                # DEBUG
                cv.rectangle(screenshot, max_location,
                             (max_location[0] + w, max_location[1] + h), (0, 0, 255), 3)

        height, width = res.shape

        for i in range(-h//2, h//2+1):
            if 0 <= max_location[1]+i < height:
                res[max_location[1]+i][max_location[0]] = 0
        for j in range(-w//2, w//+1):
            if 0 <= max_location[0]+j < width:
                res[max_location[1]][max_location[0]+j] = 0

    # DEBUG
    from matplotlib import pyplot as plt
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(screenshot)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()

    return boxes


def main():
    #by_image(None, 'original.png', second_try=True)
    result = locate_all('original.png', 1, second_try=True)
    print(result)


if __name__ == "__main__":
    main()
