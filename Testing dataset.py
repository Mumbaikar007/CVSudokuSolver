
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



image = cv2.imread( 'images/digits_sudoku4.png')

image = cv2.resize( image, None, fx=2, fy=2,)
gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)


cells = [np.hsplit(row, 39) for row in np.vsplit(gray, 9)]

# Convert the List data type to Numpy Array of shape (50,100,20,20)
x = np.array(cells)
print ("The shape of our cells array: " + str(x.shape))

# Split the full data set into two segments
# One will be used fro Training the model, the other as a test data set
train = x.astype(np.float32)  # Size = (3500,400)
# Create labels for train and test data
k = [ 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_labels = np.repeat( k, 468)[:, np.newaxis]


# Initiate kNN, train the data, then test it with test data for k=3
knn = cv2.KNearest()
knn.train(train, train_labels)
#ret, result, neighbors, distance = knn.find_nearest(test, k=3)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
'''
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size)
print("Accuracy is = %.2f" % accuracy + "%")
'''


cv2.imshow( 'Sudoku', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()