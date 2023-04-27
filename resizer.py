import cv2 as cv
import numpy as np
import os
from PIL import ImageGrab

def prepare_test_data(template):
    screenshot = ImageGrab.grab(None)
    screenshot.save('screen.png')
    screenshot = cv.imread('screen.png', cv.IMREAD_GRAYSCALE)
    s_h, s_w = screenshot.shape

    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape

    test_data=[
        [ '''
        increase
            percentage_5
                image with scale_1
                image with scale_2
                image with scale_3
            percentage_10
                image with scale_1
                image with scale_2
                image with scale_3
            ...
            percentage_50
                image with scale_1
                image with scale_2
                image with scale_3
        '''],

        #decrease
        [],

        #inc+dec
        []
    ]

    true_positions=[
        ['''
        increase
            percentage_5
                center of image with scale_1
                center of image with scale_2
                center of image with scale_3
            percentage_10
                center of image with scale_1
                center of image with scale_2
                center of image with scale_3
            ...
            percentage_50
                center of image with scale_1
                center of image with scale_2
                center of image with scale_3
        '''],

        #decrease
        [],

        #inc+dec
        []
    ]

    for percent in np.linspace(0.05, 0.5, 10):
        scales = (((int(w*(1+percent)), h), (w, int(h*(1+percent))), (int(w*(1+percent)), int(h*(1+percent)))),
                    ((int(w*(1-percent)), h), (w, int(h*(1-percent))), (int(w*(1-percent)), int(h*(1-percent)))),
                    ((int(w*(1+percent)), int(h*(1-percent))), (int(w*(1-percent)), int(h*(1+percent)))))
        p = int(percent*100)

        for scale_type in range(len(scales)): 
            p_list=[]
            pos_list=[]
            for scale in range(len(scales[scale_type])):
                # TODO merge resized & screenshot -> data, calculate center -> pos 
                data = cv.resize(template, scales[scale_type][scale])
                pos = (w//2, h//2)

                p_list.append(data)
                pos_list.append(pos)
                
            test_data[scale_type].append(p_list)
            true_positions[scale_type].append(pos_list)
            
            
    return test_data, true_positions


def prepare_resized(template, inc_path, dec_path, inc_dec_path):
    template = cv.imread(template, cv.IMREAD_GRAYSCALE)
    h, w = template.shape
    data_path=(inc_path, dec_path, inc_dec_path)

    for percent in np.linspace(0.05, 0.5, 10):
        scales = (((int(w*(1+percent)), h), (w, int(h*(1+percent))), (int(w*(1+percent)), int(h*(1+percent)))),
                    ((int(w*(1-percent)), h), (w, int(h*(1-percent))), (int(w*(1-percent)), int(h*(1-percent)))),
                    ((int(w*(1+percent)), int(h*(1-percent))), (int(w*(1-percent)), int(h*(1+percent)))))
        p = int(percent*100)

        for scale_type in range(len(scales)): 
            for scale in range(len(scales[scale_type])):
                cv.imwrite(os.path.join(data_path[scale_type], f'resized_{p}_{scale_type}_{scale}.png'), cv.resize(template, scales[scale_type][scale]))


def main():
    prepare_resized('original.png', 'test_data\increase', 'test_data\decrease', 'test_data\incdec')


if __name__ == "__main__":
    main()
