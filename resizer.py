import cv2 as cv

img = cv.imread('original.png')
print(img.shape)
w, h = img.shape[1], img.shape[0]

x = w//2
y = h//2

tmp = cv.resize(img, (w+x, h))
cv.imwrite(r'resised_1.png', tmp)

tmp = cv.resize(img, (w, h+x))
cv.imwrite(r'resised_2.png', tmp)

tmp = cv.resize(img, (w+x, h+y))
cv.imwrite(r'resised_3.png', tmp)

tmp = cv.resize(img, (w+y, h+x))
cv.imwrite(r'resised_4.png', tmp)

tmp = cv.resize(img, (w+x, h+x))
cv.imwrite(r'resised_5.png', tmp)

tmp = cv.resize(img, (w-x, h))
cv.imwrite(r'resised_6.png', tmp)

tmp = cv.resize(img, (w, h-x))
cv.imwrite(r'resised_7.png', tmp)

tmp = cv.resize(img, (w-x, h-y))
cv.imwrite(r'resised_8.png', tmp)

tmp = cv.resize(img, (w-y, h-x))
cv.imwrite(r'resised_9.png', tmp)

tmp = cv.resize(img, (w-x, h-x))
cv.imwrite(r'resised_10.png', tmp)
