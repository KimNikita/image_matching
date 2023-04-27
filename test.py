# Base
import cv2 as cv
import numpy as np
from PIL import ImageGrab
from resizer import prepare_test_data
import math
import timeit

# multi scale
import imutils

def timing(f):
    def wrapper(screenshot, template):
        start_time = timeit.default_timer()
        result = f(screenshot, template)
        ellapsed_time = timeit.default_timer() - start_time
        return result, ellapsed_time
    return wrapper


# MY TEMPLATE MATCHING
@timing
def locate_one(screenshot, template, accuracy=0.95, second_try=True, similarity=4):
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

    if similarity <= 0:
        similarity = 1
    similarity = similarity/100-0.001

    _, max_value, _, max_location = cv.minMaxLoc(res)
    w, h = template.shape[::-1]
    x, y = -1, -1

    if max_value >= accuracy:
        x = max_location[0] + w//2
        y = max_location[1] + h//2
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

            x = max_location[0] + w//2
            y = max_location[1] + h//2

    return (x, y)

# MULTI SCALE TEMPLATE MATCHING

@timing
def scale_locate_one(screenshot, template, accuracy=0.95):
    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    (h, w) = template.shape[:2]
    found_dec = (0, 0, 0)
    # decrease
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])

        if resized.shape[0] < h or resized.shape[1] < w:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_dec[0] and maxVal >= accuracy/2:
            found_dec = (maxVal, maxLoc, r)
        else:
            break

    # increase
    found_inc = (0, 0, 0)
    for scale in np.linspace(0.2, 1.0, 20):
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * (1+scale)))
        r = screenshot.shape[1] / float(resized.shape[1])

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_inc[0] and maxVal >= accuracy/2:
            found_inc = (maxVal, maxLoc, r)
        else:
            break

    if found_dec[0] > found_inc[0]:
        (maxVal, maxLoc, r) = found_dec
    else:
        (maxVal, maxLoc, r) = found_inc
    if maxVal == 0:
        return (-1, -1)

    x, y = int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2

    return (x, y)

# KEYPOINT MATCHING

@timing
def keypoint_locate_one(screenshot, template, accuracy=0.95):
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
    matches = bf.knnMatch(des1, des2, k=2)
    '''
    Short version: each keypoint of the first image is matched with a number of keypoints from the second image.
    We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement).
    Lowe's test checks that the two distances are sufficiently different.
    If they are not, then the keypoint is eliminated and will not be used for further calculations.
    https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html#:~:text=To%20filter%20the%20matches%2C%20Lowe,value%20is%20below%20a%20threshold.
    '''
    for m, n in matches:
        if m.distance < (0.7 + (1-accuracy)*0.3)*n.distance:
            best_matches.append(m)

    # Initialize lists
    list_x = []
    list_y = []
    x, y = -1, -1

    # For each match...
    for mat in best_matches:
        # Get the matching keypoints for screenshot
        screenshot_idx = mat.trainIdx

        # i - columns
        # j - rows
        # Get the coordinates
        (i, j) = kp2[screenshot_idx].pt

        # Append to each list
        list_x.append(i)
        list_y.append(j)

    if len(list_x) < 1 or len(list_y) < 1:
        return (x, y)
    elif len(list_x) > 9 and len(list_y) > 9:
        x, y = np.mean(list_x[math.floor(0.1 * len(list_x)): math.floor(-0.1 * len(list_x))]), np.mean(list_y[math.floor(0.1 * len(list_y)): math.floor(-0.1 * len(list_y))])
    else:
        x, y = np.mean(list_x), np.mean(list_y) 

    return (x, y)


def main():
    test_data, true_positions, max_distances = prepare_test_data('template.png')

    accuracy_results = [
        # my_locate_one
        [
            #increase
                #percentage_5
                    #mean accuracy
                #percentage_10
                    #mean accuracy
                #..................
                #percentage_50
                    #mean accuracy
            [],

            # decrease
            [],

            # inc+dec
            []
        ],

        # scale_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ]
    ]

    time_results = [
        # my_locate_one
        [
            #increase
                #percentage_5
                    #mean time
                #percentage_10
                    #mean time
                #..................
                #percentage_50
                    #mean time
            [],

            # decrease
            [],

            # inc+dec
            []
        ],

        # scale_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ]
    ]

    point_results = [
        # my_locate_one
        [
            [],

            [],

            []
        ],

        # scale_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            []
        ]
    ]

    i = -1
    for alg in (locate_one, scale_locate_one, keypoint_locate_one):
        i += 1
        print(f'Stage: {i}')
        for scale_type in range(len(test_data)):
            for percentage in range(len(test_data[scale_type])):
                p_res=[]
                a_res = []
                times = []
                for screenshot in range(len(test_data[scale_type][percentage])):
                    test_screenshot = test_data[scale_type][percentage][screenshot].copy()
                    
                    # TODO fix not finding
                    point_res, time = alg(test_screenshot, 'template.png')
                    p_res.append(point_res)
                    times.append(time)

                    accuracy = 1
                    if point_res[0] < 0 or point_res[1] < 0:
                        # алгоритм не нашел
                        accuracy = -1
                    else:
                        # ВАЖНО!!! если accuracy > 0 значит алгоритм в любом случае попадет по контролу, насколько accuracy близко к 1 не так важно 
                        accuracy = 1 - math.dist(true_positions[scale_type][percentage][screenshot], point_res) / max_distances[scale_type][percentage][screenshot] 
                        # алгоритм промахнулся
                        if accuracy < 0:
                            accuracy = 0
                    a_res.append(accuracy)

                # DEBUG
                # time_results[i][scale_type].append(times)
                # accuracy_results[i][scale_type].append(a_res)
                # point_results[i][scale_type].append(p_res)

                # RELEASE
                time_results[i][scale_type].append(np.mean(times))
                accuracy_results[i][scale_type].append(np.mean(a_res))

    # DEBUG
    print(point_results)
    print()
    print(accuracy_results)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet()

    # increase table
    worksheet.write(0, 0, '"%" increase')
    worksheet.write(0, 1, 'my_template_matching')
    worksheet.write(0, 2, 'multi-scale_template_matching')
    worksheet.write(0, 3, 'keypoint_matching')
    worksheet.write(0, 4, 'times')
    worksheet.write(0, 5, 'my_template_matching')
    worksheet.write(0, 6, 'multi-scale_template_matching')
    worksheet.write(0, 7, 'keypoint_matching')
    
    row = 0
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row+=1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[0][row-1])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+5, alg[0][row-1])

    # decrease table
    worksheet.write(11, 0, '"%" decrease')
    worksheet.write(11, 1, 'my_template_matching')
    worksheet.write(11, 2, 'multi-scale_template_matching')
    worksheet.write(11, 3, 'keypoint_matching')
    worksheet.write(11, 4, 'times')
    worksheet.write(11, 5, 'my_template_matching')
    worksheet.write(11, 6, 'multi-scale_template_matching')
    worksheet.write(11, 7, 'keypoint_matching')

    row = 11
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row+=1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[1][row-12])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+5, alg[1][row-12])

    # incdec table
    worksheet.write(22, 0, '"%" incdec')
    worksheet.write(22, 1, 'my_template_matching')
    worksheet.write(22, 2, 'multi-scale_template_matching')
    worksheet.write(22, 3, 'keypoint_matching')
    worksheet.write(22, 4, 'times')
    worksheet.write(22, 5, 'my_template_matching')
    worksheet.write(22, 6, 'multi-scale_template_matching')
    worksheet.write(22, 7, 'keypoint_matching')

    row = 22
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row+=1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[2][row-23])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+5, alg[2][row-23])

    
    workbook.close()

if __name__ == "__main__":
    main()
