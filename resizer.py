import cv2 as cv
import numpy as np
import os
import imutils
import random
import glob
from PIL import ImageGrab

# percentage distribution
def prepare_test_data_0(template):
    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)
    s_h, s_w = screenshot.shape

    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape

    screenshot = screenshot[int(s_h//2-1.5*h): int(s_h//2+1.5*h), int(s_w//2-1.5*w): int(s_w//2+1.5*w)]
    s_h, s_w = screenshot.shape

    test_data = [
        # increase
            # percentage_5
                # image with scale_1
                # image with scale_2
                # image with scale_3
            # percentage_10
                # image with scale_1
                # image with scale_2
                # image with scale_3
            # ...
            # percentage_50
                # image with scale_1
                # image with scale_2
                # image with scale_3
        [],

        # decrease
        [],

        # inc+dec
        [],

        []

    ]

    true_positions = [
        # increase
            # percentage_5
                # center of image with scale_1
                # center of image with scale_2
                # center of image with scale_3
            # percentage_10
                # center of image with scale_1
                # center of image with scale_2
                # center of image with scale_3
            # ...
            # percentage_50
                # center of image with scale_1
                # center of image with scale_2
                # center of image with scale_3
        [],

        # decrease
        [],

        # inc+dec
        [],

        []
    ]

    max_distances = [
        [],
        [],
        [],
        []
    ]

    for percent in np.linspace(0.05, 0.5, 10):
        scales = (((int(w*(1+percent)), h), (w, int(h*(1+percent))), (int(w*(1+percent)), int(h*(1+percent)))),
                    ((int(w*(1-percent)), h), (w, int(h*(1-percent))),
                     (int(w*(1-percent)), int(h*(1-percent)))),
                    ((int(w*(1+percent)), int(h*(1-percent))), (int(w*(1-percent)), int(h*(1+percent)))))
        proportional_scales = (int(w*(1+percent)), int(w*(1-percent)))

        p = int(percent*100)

        for scale_type in range(len(scales)):
            p_list = []
            pos_list = []
            dist_list = []
            for scale in range(len(scales[scale_type])):
                resized = cv.resize(template, scales[scale_type][scale])
                r_h, r_w = resized.shape
                dist_list.append(min(r_w//2, r_h//2))

                data = screenshot.copy()
                pos = (s_w//2, s_h//2)

                x_offset = s_w//2 - r_w//2
                y_offset = s_h//2 - r_h//2
                data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized

                p_list.append(data)
                pos_list.append(pos)

            max_distances[scale_type].append(dist_list)
            test_data[scale_type].append(p_list)
            true_positions[scale_type].append(pos_list)

        p_list = []
        pos_list = []
        dist_list = []
        for scale in range(len(proportional_scales)):
            resized = imutils.resize(
                template, width=proportional_scales[scale])

            r_h, r_w = resized.shape
            dist_list.append(min(r_w//2, r_h//2))

            data = screenshot.copy()
            pos = (s_w//2, s_h//2)

            x_offset = s_w//2 - r_w//2
            y_offset = s_h//2 - r_h//2
            data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized

            p_list.append(data)
            pos_list.append(pos)

        max_distances[3].append(dist_list)
        test_data[3].append(p_list)
        true_positions[3].append(pos_list)

    return test_data, true_positions, max_distances

# finding position of control
def prepare_test_data_1(template):
    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)
    s_h, s_w = screenshot.shape

    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape

    screenshot = screenshot[int(s_h//2-1.5*h): int(s_h//2+1.5*h), int(s_w//2-1.5*w): int(s_w//2+1.5*w)]
    s_h, s_w = screenshot.shape

    test_data = []

    true_positions = []

    max_distances = []

    scales = []
    proportional_scales = []

    for percent in np.linspace(0.05, 0.5, 10):
        scales.append((int(w*(1+percent)), h))
        scales.append((w, int(h*(1+percent))))
        scales.append((int(w*(1+percent)), int(h*(1+percent))))
        scales.append((int(w*(1-percent)), h))
        scales.append((w, int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1-percent))))
        scales.append((int(w*(1+percent)), int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1+percent))))
        proportional_scales.append(int(w*(1+percent)))
        proportional_scales.append(int(w*(1-percent)))

    random.shuffle(scales)
    random.shuffle(proportional_scales)

    for scale in range(len(scales)//2):
        resized = cv.resize(template, scales[scale])
        r_h, r_w = resized.shape
        max_distances.append(min(r_w//2, r_h//2))

        data = screenshot.copy()
        pos = (s_w//2, s_h//2)
        x_offset = s_w//2 - r_w//2
        y_offset = s_h//2 - r_h//2
        data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized

        test_data.append(data)
        true_positions.append(pos)

    for scale in range(len(proportional_scales)//2): 
        resized = imutils.resize(template, width=proportional_scales[scale])
        r_h, r_w = resized.shape
        max_distances.append(min(r_w//2, r_h//2))
        
        data = screenshot.copy()
        pos = (s_w//2, s_h//2)
        x_offset = s_w//2 - r_w//2
        y_offset = s_h//2 - r_h//2
        data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized
            
        test_data.append(data)
        true_positions.append(pos)

        
    return test_data, true_positions, max_distances

# finding positions of multiple controls
def prepare_test_data_2(template):
    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)
    s_h, s_w = screenshot.shape

    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape

    screenshot = screenshot[max(0,int(s_h//2-3*h)): min(s_h,int(s_h//2+4*h)), max(0,int(s_w//2-4*w)): min(s_w,int(s_w//2+4*w))]
    s_h, s_w = screenshot.shape

    test_data = []

    true_positions = []

    max_distances = []

    scales = []

    for percent in np.linspace(0.05, 0.5, 10):
        scales.append((int(w*(1+percent)), h))
        scales.append((w, int(h*(1+percent))))
        scales.append((int(w*(1+percent)), int(h*(1+percent))))
        scales.append((int(w*(1-percent)), h))
        scales.append((w, int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1-percent))))
        scales.append((int(w*(1+percent)), int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1+percent))))

    random.shuffle(scales)
    for scale in range(0, len(scales)//2-4, 5):
        resized_all=[]
        dists=[]
        shapes=[]
        for i in range(5):
            resized = cv.resize(template, scales[scale+i])
            r_h, r_w = resized.shape
            shapes.append((r_w, r_h))
            dists.append(min(r_w//2, r_h//2))
            resized_all.append(resized)

        max_distances.append(dists)

        data = screenshot.copy()
        pos = [(s_w//2-2*w, s_h//2-h*3//2), (s_w//2, s_h//2-h*3//2), (s_w//2+2*w, s_h//2-h*3//2), (s_w//2-w, s_h//2+h*3//2), (s_w//2+w, s_h//2+h*3//2)]
        
        for i in range(5):
            x_offset = pos[i][0] - shapes[i][0]//2
            y_offset = pos[i][1] - shapes[i][1]//2
            data[y_offset:y_offset+shapes[i][1], x_offset:x_offset+shapes[i][0]] = resized_all[i]

        test_data.append(data)
        true_positions.append(pos)

    return test_data, true_positions, max_distances

# confirming absence of control
def prepare_test_data_3(template):
    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)
    s_h, s_w = screenshot.shape

    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape

    test_data = []

    scales = []
    proportional_scales = []

    for percent in np.linspace(0.05, 0.5, 10):
        scales.append((int(w*(1+percent)), h))
        scales.append((w, int(h*(1+percent))))
        scales.append((int(w*(1+percent)), int(h*(1+percent))))
        scales.append((int(w*(1-percent)), h))
        scales.append((w, int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1-percent))))
        scales.append((int(w*(1+percent)), int(h*(1-percent))))
        scales.append((int(w*(1-percent)), int(h*(1+percent))))
        proportional_scales.append(int(w*(1+percent)))
        proportional_scales.append(int(w*(1-percent)))

    random.shuffle(scales)
    random.shuffle(proportional_scales)

    for scale in range(len(scales)//2):
        resized = cv.resize(template, scales[scale])
        r_h, r_w = resized.shape

        data = screenshot.copy()
        pos = [ (s_w//2-r_w, s_h//2-r_h), (s_w//2, s_h//2-r_h), (s_w//2+r_w, s_h//2-r_h), 
                (s_w//2-r_w, s_h//2),                           (s_w//2+r_w, s_h//2),
                (s_w//2-r_w, s_h//2+r_h), (s_w//2, s_h//2+r_h), (s_w//2+r_w, s_h//2+r_h)]

        for i in range(8):
            x_offset = pos[i][0] - r_w//2
            y_offset = pos[i][1] - r_h//2
            data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized

        actual_data = data[s_h//2-r_h*5//4:s_h//2+r_h*5//4, s_w//2-r_w*5//4:s_w//2+r_w*5//4]
        test_data.append(actual_data)

    for scale in range(len(proportional_scales)//2): 
        resized = imutils.resize(template, width=proportional_scales[scale])
        r_h, r_w = resized.shape
        
        data = screenshot.copy()
        pos = [ (s_w//2-r_w, s_h//2-r_h), (s_w//2, s_h//2-r_h), (s_w//2+r_w, s_h//2-r_h), 
                (s_w//2-r_w, s_h//2),                           (s_w//2+r_w, s_h//2),
                (s_w//2-r_w, s_h//2+r_h), (s_w//2, s_h//2+r_h), (s_w//2+r_w, s_h//2+r_h)]

        for i in range(8):
            x_offset = pos[i][0] - r_w//2
            y_offset = pos[i][1] - r_h//2
            data[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = resized

        actual_data = data[s_h//2-r_h*5//4:s_h//2+r_h*5//4, s_w//2-r_w*5//4:s_w//2+r_w*5//4]
        test_data.append(actual_data)
        
    return test_data

def prepare_test_data_4(screenshots_path, templates_path):
    #test_data = prepare_test_data_4('real_test_screenshots', 'real_test_templates')
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
    test_data = [
        [],
        [],
        [],
        [],
        []
    ]

    # from .txt
    positions = [
        [
            (30, 36), (30, 36), (34, 38), (30, 36)
        ],

        [
            (979, 24), (1059, 24), (1538, 22), (419, 23)
        ],

        [
            (1094, 22), (1172, 16), (1652, 20), (532, 18)
        ],

        [
           (78, 38), (74, 36), (72, 38),  (72, 34)
        ],

        [
            (22, 20), (22, 20), (22, 22), (22, 18)
        ]
    ]

    for i in range(5):
        for template in glob.glob(os.path.join(templates_path, str(i+1), '*.png')):
            template_data = cv.imread(template, cv.IMREAD_GRAYSCALE)
            h, w = template_data.shape
            j=-1
            for screenshot in glob.glob(os.path.join(screenshots_path, str(i+1), '*.png')):
                j+=1
                screenshot_data = cv.imread(screenshot, cv.IMREAD_GRAYSCALE)
                test_data[i].append([screenshot_data, template_data, positions[i][j], min(w//2, h//2)])

    return test_data


def prepare_resized_data(template, inc_path, dec_path, inc_dec_path, proportional_path):
    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape
    data_path=(inc_path, dec_path, inc_dec_path, proportional_path)

    for percent in np.linspace(0.05, 0.5, 10):
        scales = (((int(w*(1+percent)), h), (w, int(h*(1+percent))), (int(w*(1+percent)), int(h*(1+percent)))),
                    ((int(w*(1-percent)), h), (w, int(h*(1-percent))), (int(w*(1-percent)), int(h*(1-percent)))),
                    ((int(w*(1+percent)), int(h*(1-percent))), (int(w*(1-percent)), int(h*(1+percent)))))
        proportional_scales = (int(w*(1+percent)), int(w*(1-percent)))
        
        p = int(percent*100)

        for scale_type in range(len(scales)): 
            for scale in range(len(scales[scale_type])):
                cv.imwrite(os.path.join(data_path[scale_type], f'resized_{p}_{scale_type}_{scale}.png'), cv.resize(template, scales[scale_type][scale]))
        for scale in range(len(proportional_scales)):
                cv.imwrite(os.path.join(proportional_path, f'resized_{p}_3_{scale}.png'), imutils.resize(template, width=proportional_scales[scale]))


def main():
    prepare_resized_data('template.png', 'test_data\increase', 'test_data\decrease', 'test_data\incdec', 'test_data\proportional')

if __name__ == "__main__":
    main()
