
import cv2
import numpy as np

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed

    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        # print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = (height - width) / 2
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, \
                                                   pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width - height) / 2
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, \
                                                   cv2.BORDER_CONSTANT, value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    # print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square


def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions

    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg



image = cv2.imread( 'images/digits_sudoku.png')

gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur( gray, (5,5), 0 )

threshed = cv2.adaptiveThreshold( blur,255,1,1,11,2)

contours, hierarchy = cv2.findContours( threshed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

print(len(contours))

for c in contours:

    (x, y, w, h) = cv2.boundingRect(c)

    roi = blur[y:y + h, x:x + w]
    ret, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
    squared = makeSquare(roi)
    final = resize_to_pixel(20, squared)

    cv2.imshow('sq', final)
    final_array = final.reshape((1, 400))
    final_array = final_array.astype(np.float32)
    #ret, result, neighbours, dist = knn.find_nearest(final_array, k=1)
    #number = str(int(float(result[0])))
    #full_number.append(number)
    # draw a rectangle around the digit, the show what the
    # digit was classified as
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.putText(image, number, (x, y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

cells = [np.hsplit(row, 5) for row in np.vsplit(gray, 34)]

cv2.imshow( 'Sudoku', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()