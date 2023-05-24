# all
import cv2 as cv
import numpy as np
from PIL import ImageGrab
from resizer import prepare_test_data_0, prepare_test_data_1, prepare_test_data_2, prepare_test_data_3, prepare_test_data_4
import math
import timeit
import operator

# multi scale
import imutils

# keypoint
from sklearn.cluster import KMeans

def timing(f):
    def wrapper(screenshot, template):
        start_time = timeit.default_timer()
        result = f(screenshot, template)
        ellapsed_time = timeit.default_timer() - start_time
        return result, ellapsed_time
    return wrapper

# ------------------- LOCATE ONE ------------------------

# BASE TEMPLATE MATCHING
@timing
def base_locate_one(screenshot, template, accuracy=0.95):
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

    w, h = template.shape[::-1]

    x, y = -1, -1
    if max_value >= accuracy:
        x = max_location[0] + w//2
        y = max_location[1] + h//2

    return (x, y)

# MY TEMPLATE MATCHING


@timing
def my_locate_one(screenshot, template, accuracy=0.95, second_try=True):
    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None

    # Apply template Matching
    res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)

    _, max_value, _, max_location = cv.minMaxLoc(res)

    h, w = template.shape

    x, y = -1, -1

    if max_value >= accuracy:
        x = max_location[0] + w//2
        y = max_location[1] + h//2
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

    h, w = template.shape
    found_dec = (0, 0, 0)
    prev_max= [0, 0]
    next_min= [0, 0]
    # decrease
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * scale))
        r = screenshot.shape[1] / float(resized.shape[1])

        if resized.shape[0] < h or resized.shape[1] < w:
            break

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        if maxVal > found_dec[0]:
            prev_max[0] = found_dec[0]
            found_dec = (maxVal, maxLoc, r)
        else:
            next_min[0]=maxVal
            break

    # increase
    found_inc = (0, 0, 0)
    for scale in np.linspace(0.04, 0.8, 19):
        resized = imutils.resize(
            screenshot, width=int(screenshot.shape[1] * (1+scale)))
        r = screenshot.shape[1] / float(resized.shape[1])

        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

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

    w, h = template.shape[::-1]

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    if des1 is None or des2 is None:
        return (-1, -1)

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
   
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < (0.7 + (1-accuracy)*0.3)*n.distance:
            best_matches.append(m)

    if len(best_matches)==0:
        return (-1, -1)

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
        x, y = np.mean(sorted(list_x)[math.ceil(0.1 * len(list_x)): math.ceil(-0.1 * len(list_x))]), np.mean(sorted(list_y)[math.ceil(0.1 * len(list_y)): math.ceil(-0.1 * len(list_y))])
    else:
        x, y = np.mean(list_x), np.mean(list_y)

    return (x, y)

# ADVANCED TEMPLATE MATCHING

@timing
def advanced_locate_one(screenshot, template, accuracy=0.95, second_try=True):
    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None

    h, w = template.shape
    x, y = -1, -1

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
       x, y = int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2
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

    return (x, y)


# ------------------- LOCATE All ------------------------

@timing
def my_locate_all(screenshot, template, accuracy=0.95, second_try=True):
    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    points=[]
    height, width = screenshot.shape
    for i in range(5):
        # Apply template Matching
        res = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)

        _, max_value, _, max_location = cv.minMaxLoc(res)

        h, w = template.shape

        x, y = -1, -1

        if max_value >= accuracy:
            x = max_location[0] + w//2
            y = max_location[1] + h//2
        elif second_try:
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
                x = max_location[0] + w//2
                y = max_location[1] + h//2
        
        points.append((x, y))

        if x != -1 and y != -1:
            for i in range(0, h):
                for j in range(0, w):
                    if 0 <= max_location[1]+i < height and 0 <= max_location[0]+j < width:
                        screenshot[max_location[1]+i][max_location[0]+j] = 0

    return points

# MULTI SCALE TEMPLATE MATCHING


@timing
def scale_locate_all(screenshot, template, accuracy=0.95):
    if isinstance(template, str):
        template = cv.imread(template, cv.IMREAD_GRAYSCALE)
        if template is None:
            print('Cannot read image, check cv2.imread() documentation')
            return None
    else:
        print('Invalid format of image')
        return None

    h, w = template.shape
    points=[]
    height, width = screenshot.shape
    for j in range(5):
        found_dec = (0, 0, 0)
        prev_max= [0, 0]
        next_min= [0, 0]
        # decrease
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(
                screenshot, width=int(screenshot.shape[1] * scale))
            r = screenshot.shape[1] / float(resized.shape[1])

            if resized.shape[0] < h or resized.shape[1] < w:
                break

            result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

            if maxVal > found_dec[0]:
                prev_max[0] = found_dec[0]
                found_dec = (maxVal, maxLoc, r)
            else:
                next_min[0]=maxVal
                break

        # increase
        found_inc = (0, 0, 0)
        for scale in np.linspace(0.04, 0.8, 19):
            resized = imutils.resize(
                screenshot, width=int(screenshot.shape[1] * (1+scale)))
            r = screenshot.shape[1] / float(resized.shape[1])

            result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

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
            points.append((-1, -1))
            continue

        x, y = int(maxLoc[0] * r) + w//2, int(maxLoc[1] * r) + h//2
        points.append((x, y))

        for i in range(0, h):
            for j in range(0, w):
                if 0 <= int(maxLoc[1] * r)+i < height and 0 <= int(maxLoc[0] * r)+j < width:
                    screenshot[int(maxLoc[1] * r)+i][int(maxLoc[0] * r)+j] = 0

    return points

# KEYPOINT MATCHING


@timing
def keypoint_locate_all(screenshot, template, accuracy=0.95):
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

    if des1 is None or des2 is None:
        return [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]

    # BFMatcher
    best_matches = []

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=6)
    '''
    Short version: each keypoint of the first image is matched with a number of keypoints from the second image.
    We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement).
    Lowe's test checks that the two distances are sufficiently different.
    If they are not, then the keypoint is eliminated and will not be used for further calculations.
    https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html#:~:text=To%20filter%20the%20matches%2C%20Lowe,value%20is%20below%20a%20threshold.
    '''
   
    for points in matches:
        if len(points)<6:
            continue
        for i in range(5):
            if points[i].distance < (0.7 + (1-accuracy)*0.3)*points[5].distance:
                best_matches.append(points[i])

    if len(best_matches)==0:
        return [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]

    points=[]
    # For each match...
    for mat in best_matches:
        # Get the matching keypoints for screenshot
        screenshot_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x, y) = kp2[screenshot_idx].pt

        # Append to each list
        points.append([x, y])

    clusters = KMeans(n_clusters=min(5, len(points)), n_init="auto").fit(points)

    points=[]
    for point in clusters.cluster_centers_:
        points.append((point[0], point[1]))

    return points


# percentage distribution
def test_0():
    test_data, true_positions, max_distances = prepare_test_data_0('template.png')

    accuracy_results = [
        # base template matching
        [
            [],
            [],
            [],
            []
        ],

        # my_locate_one
        [
            # increase
                # percentage_5
                    # mean accuracy
                # percentage_10
                    # mean accuracy
                # ..................
                # percentage_50
                    # mean accuracy
            [],

            # decrease
            [],

            # inc+dec
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
            [],

            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            [],

            []
        ]
    ]

    time_results = [
        # base template matching
        [
            [],
            [],
            [],
            []
        ],
        # my_locate_one
        [
            # increase
            # percentage_5
            # mean time
            # percentage_10
            # mean time
            # ..................
            # percentage_50
            # mean time
            [],

            # decrease
            [],

            # inc+dec
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
            [],

            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            [],

            []
        ]
    ]

    point_results = [
        # base template matching
        [
            [],
            [],
            [],
            []
        ],
        # my_locate_one
        [
            [],

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
            [],

            []
        ],

        # keypoint_locate_one
        [
            # increase
            [],

            # decrease
            [],

            # inc+dec
            [],

            []
        ]
    ]

    i = -1
    for alg in (base_locate_one, my_locate_one, scale_locate_one, keypoint_locate_one):
        i += 1
        print(f'Stage: {i}')
        for scale_type in range(len(test_data)):
            for percentage in range(len(test_data[scale_type])):
                p_res = []
                a_res = []
                times = []
                for screenshot in range(len(test_data[scale_type][percentage])):
                    test_screenshot = test_data[scale_type][percentage][screenshot].copy()

                    point_res, time = alg(test_screenshot, 'template.png')
                    p_res.append(point_res)
                    times.append(time)

                    accuracy = 1
                    if point_res[0] < 0 or point_res[1] < 0:
                        # алгоритм не нашел (ошибка 1 рода)
                        accuracy = 0
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
                point_results[i][scale_type].append(p_res)

    # DEBUG
    print(point_results)
    print()
    print(accuracy_results)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_0.xlsx')
    worksheet = workbook.add_worksheet()

    # increase table
    worksheet.write(0, 0, '"%" increase')
    worksheet.write(0, 1, 'base_template_matching')
    worksheet.write(0, 2, 'my_template_matching')
    worksheet.write(0, 3, 'multi-scale_template_matching')
    worksheet.write(0, 4, 'keypoint_matching')
    worksheet.write(0, 5, 'times')
    worksheet.write(0, 6, 'base_template_matching')
    worksheet.write(0, 7, 'my_template_matching')
    worksheet.write(0, 8, 'multi-scale_template_matching')
    worksheet.write(0, 9, 'keypoint_matching')

    row = 0
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row += 1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[0][row-1])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+6, alg[0][row-1])

    # decrease table
    worksheet.write(11, 0, '"%" decrease')
    worksheet.write(11, 1, 'base_template_matching')
    worksheet.write(11, 2, 'my_template_matching')
    worksheet.write(11, 3, 'multi-scale_template_matching')
    worksheet.write(11, 4, 'keypoint_matching')
    worksheet.write(11, 5, 'times')
    worksheet.write(11, 6, 'base_template_matching')
    worksheet.write(11, 7, 'my_template_matching')
    worksheet.write(11, 8, 'multi-scale_template_matching')
    worksheet.write(11, 9, 'keypoint_matching')

    row = 11
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row += 1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[1][row-12])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+6, alg[1][row-12])

    # incdec table
    worksheet.write(22, 0, '"%" incdec')
    worksheet.write(22, 1, 'base_template_matching')
    worksheet.write(22, 2, 'my_template_matching')
    worksheet.write(22, 3, 'multi-scale_template_matching')
    worksheet.write(22, 4, 'keypoint_matching')
    worksheet.write(22, 5, 'times')
    worksheet.write(22, 6, 'base_template_matching')
    worksheet.write(22, 7, 'my_template_matching')
    worksheet.write(22, 8, 'multi-scale_template_matching')
    worksheet.write(22, 9, 'keypoint_matching')

    row = 22
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row += 1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[2][row-23])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+6, alg[2][row-23])

    # proportional table
    worksheet.write(33, 0, '"%" proportional_resize')
    worksheet.write(33, 1, 'base_template_matching')
    worksheet.write(33, 2, 'my_template_matching')
    worksheet.write(33, 3, 'multi-scale_template_matching')
    worksheet.write(33, 4, 'keypoint_matching')
    worksheet.write(33, 5, 'times')
    worksheet.write(33, 6, 'base_template_matching')
    worksheet.write(33, 7, 'my_template_matching')
    worksheet.write(33, 8, 'multi-scale_template_matching')
    worksheet.write(33, 9, 'keypoint_matching')

    row = 33
    for percent in np.linspace(0.05, 0.5, 10):
        p = int(percent*100)
        row += 1
        worksheet.write(row, 0, p)
        for col, alg in enumerate(accuracy_results):
            worksheet.write(row, col+1, alg[3][row-34])
        for col, alg in enumerate(time_results):
            worksheet.write(row, col+6, alg[3][row-34])

    workbook.close()

# finding position of control
def test_1():
    test_data, true_positions, max_distances = prepare_test_data_1('template.png')

    accuracy_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]

    time_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]

    i = -1
    for alg in (my_locate_one, scale_locate_one, keypoint_locate_one):
        i += 1
        print(f'Stage: {i}')
        for screenshot in range(len(test_data)):
            test_screenshot = test_data[screenshot].copy()
            point_res, time = alg(test_screenshot, 'template.png')
            time_results[i].append(time)

            accuracy = 1
            if point_res[0] < 0 or point_res[1] < 0:
                # алгоритм не нашел (ошибка 1 рода)
                accuracy = 0
            else:
                # ВАЖНО!!! если accuracy > 0 значит алгоритм в любом случае попадет по контролу, насколько accuracy близко к 1 не так важно
                accuracy = 1 - math.dist(true_positions[screenshot], point_res) / max_distances[screenshot]
                # алгоритм промахнулся
                if accuracy < 0:
                    accuracy = 0
            accuracy_results[i].append(accuracy)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_1.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 1, 'my_template_matching')
    worksheet.write(0, 2, 'multi-scale_template_matching')
    worksheet.write(0, 3, 'keypoint_matching')

    worksheet.write(1, 0, 'Mean accuracy')
    worksheet.write(2, 0, 'Mean time')


    for col, alg in enumerate(accuracy_results):
        worksheet.write(1, col+1, np.mean(alg))

    for col, alg in enumerate(time_results):
        worksheet.write(2, col+1, np.mean(alg))

    workbook.close()

# finding positions of multiple controls
def test_2():
    test_data, true_positions, max_distances = prepare_test_data_2('template.png')

    accuracy_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]

    time_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]
    
    i = -1
    for alg in (my_locate_all, scale_locate_all, keypoint_locate_all):
        i += 1
        print(f'Stage: {i}')
        for screenshot in range(len(test_data)):
            test_screenshot = test_data[screenshot].copy()
            points, time = alg(test_screenshot, 'template.png')

            time_results[i].append(time)

            mean_accuracy=[]
            for j in range(min(5, len(points))):
                best_match_accuracy = 1
                if points[j][0] < 0 or points[j][1] < 0:
                    # алгоритм не нашел (ошибка 1 рода)
                    best_match_accuracy = 0
                else:
                    best_match_accuracy = 0
                    for k in range(5):
                        # ВАЖНО!!! если accuracy > 0 значит алгоритм в любом случае попадет по контролу, насколько accuracy близко к 1 не так важно
                        accuracy = 1 - math.dist(true_positions[screenshot][k], points[j]) / max_distances[screenshot][k]
                        # алгоритм промахнулся
                        if accuracy > best_match_accuracy:
                            best_match_accuracy = accuracy
                mean_accuracy.append(best_match_accuracy)
            accuracy_results[i].append(np.mean(mean_accuracy))

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_2.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 1, 'my_template_matching')
    worksheet.write(0, 2, 'multi-scale_template_matching')
    worksheet.write(0, 3, 'keypoint_matching')

    worksheet.write(1, 0, 'Mean accuracy')
    worksheet.write(2, 0, 'Mean time')


    for col, alg in enumerate(accuracy_results):
        worksheet.write(1, col+1, np.mean(alg))

    for col, alg in enumerate(time_results):
        worksheet.write(2, col+1, np.mean(alg))

    workbook.close()

# confirming absence of control
def test_3():
    test_data = prepare_test_data_3('template.png')

    accuracy_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]

    time_results = [
        # my_locate_one
        [],

        # scale_locate_one
        [],

        # keypoint_locate_one
        []
    ]


    i = -1
    for alg in (my_locate_one, scale_locate_one, keypoint_locate_one):
        i += 1
        print(f'Stage: {i}')
        for screenshot in range(len(test_data)):
            test_screenshot = test_data[screenshot].copy()
            point_res, time = alg(test_screenshot, 'template.png')
            time_results[i].append(time)

            if point_res[0] < 0 or point_res[1] < 0:
                accuracy_results[i].append(1)
            else:
                accuracy_results[i].append(0)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_3.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 1, 'my_template_matching')
    worksheet.write(0, 2, 'multi-scale_template_matching')
    worksheet.write(0, 3, 'keypoint_matching')

    worksheet.write(1, 0, 'Mean accuracy')
    worksheet.write(2, 0, 'Mean time')


    for col, alg in enumerate(accuracy_results):
        worksheet.write(1, col+1, np.mean(alg))

    for col, alg in enumerate(time_results):
        worksheet.write(2, col+1, np.mean(alg))

    workbook.close()

# real data
def test_4():
    test_data = prepare_test_data_4('real_test_screenshots', 'real_test_templates')

    # test_data = 
        # control_1
        # [
            # (screenshot, template, pos, dist)
            # (screenshot, template, pos, dist)
            # ......................
            # (screenshot, template, pos, dist)
        # ]
        # control_2
        # .........
        # control_5

    accuracy_results = [
        # my_locate_one
        [[],[],[],[],[]],

        # scale_locate_one
        [[],[],[],[],[]],

        # keypoint_locate_one
        [[],[],[],[],[]]
    ]

    time_results = [
        # my_locate_one
        [[],[],[],[],[]],

        # scale_locate_one
        [[],[],[],[],[]],

        # keypoint_locate_one
        [[],[],[],[],[]]
    ]


    i = -1
    for alg in (my_locate_one, scale_locate_one, keypoint_locate_one):
        i += 1
        print(f'Stage: {i}')
        j=-1
        for control in test_data:
            j+=1
            for data in control:
                test_screenshot = data[0].copy()
                test_template = data[1].copy()
                point_res, time = alg(test_screenshot, test_template)
                time_results[i][j].append(time)

                accuracy = 1
                #print(data[2], point_res)
                if point_res[0] < 0 or point_res[1] < 0:
                    accuracy = 0
                else:
                    accuracy = 1 - math.dist(data[2], point_res) / data[3]
                if accuracy < 0:
                    accuracy = 0
                accuracy_results[i][j].append(accuracy)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_4.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 1, 'CONTROL 1')
    worksheet.write(0, 5, 'CONTROL 2')
    worksheet.write(0, 9, 'CONTROL 3')
    worksheet.write(0, 13, 'CONTROL 4')
    worksheet.write(0, 17, 'CONTROL 5')


    for alg_col, alg in enumerate(accuracy_results):
        for control_col, control in enumerate(alg):
            for res_row, res in enumerate(control):
                worksheet.write(1+res_row*3, control_col*4+alg_col, res)

    for alg_col, alg in enumerate(time_results):
        for control_col, control in enumerate(alg):
            for res_row, res in enumerate(control):
                worksheet.write(2+res_row*3, control_col*4+alg_col, res)

    workbook.close()

# real data
def test_5():
    test_data = prepare_test_data_4('real_test_screenshots', 'real_test_templates')

    # test_data = 
        # control_1
        # [
            # (screenshot, template, pos, dist)
            # (screenshot, template, pos, dist)
            # ......................
            # (screenshot, template, pos, dist)
        # ]
        # control_2
        # .........
        # control_5

    accuracy_results = [
        # my_locate_one
        [[],[],[],[],[]],

        # scale_locate_one
        [[],[],[],[],[]],

        # advanced_locate_one
        [[],[],[],[],[]]
    ]

    time_results = [
        # my_locate_one
        [[],[],[],[],[]],

        # scale_locate_one
        [[],[],[],[],[]],

        # advanced_locate_one
        [[],[],[],[],[]]
    ]


    i = -1
    for alg in (my_locate_one, scale_locate_one, advanced_locate_one):
        i += 1
        print(f'Stage: {i}')
        j=-1
        for control in test_data:
            j+=1
            for data in control:
                test_screenshot = data[0].copy()
                test_template = data[1].copy()
                point_res, time = alg(test_screenshot, test_template)
                time_results[i][j].append(time)

                accuracy = 1
                if point_res[0] < 0 or point_res[1] < 0:
                    accuracy = 0
                else:
                    accuracy = 1 - math.dist(data[2], point_res) / data[3]
                if accuracy < 0:
                    accuracy = 0
                accuracy_results[i][j].append(accuracy)

    import xlsxwriter

    workbook = xlsxwriter.Workbook('results_of_test_5.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 1, 'CONTROL 1')
    worksheet.write(0, 5, 'CONTROL 2')
    worksheet.write(0, 9, 'CONTROL 3')
    worksheet.write(0, 13, 'CONTROL 4')
    worksheet.write(0, 17, 'CONTROL 5')


    for alg_col, alg in enumerate(accuracy_results):
        for control_col, control in enumerate(alg):
            for res_row, res in enumerate(control):
                worksheet.write(1+res_row*3, control_col*4+alg_col, res)

    for alg_col, alg in enumerate(time_results):
        for control_col, control in enumerate(alg):
            for res_row, res in enumerate(control):
                worksheet.write(2+res_row*3, control_col*4+alg_col, res)

    workbook.close()
    

def main():
    #print('---------------------------TEST 0---------------------------')
    #test_0() 

    #print('---------------------------TEST 1---------------------------')
    #test_1()

    #print('---------------------------TEST 2---------------------------')
    #test_2()

    #print('---------------------------TEST 3---------------------------')
    #test_3()

    #print('---------------------------TEST 4---------------------------')
    #test_4()

    print('---------------------------TEST 5---------------------------')
    test_5()


if __name__ == "__main__":
    main()
