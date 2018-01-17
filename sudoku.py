
import cv2
import numpy as np

def shrink_image ( image ):

    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    r = 500.0 / image.shape[1]
    dim = (500, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def wrapper ( image, tl, tr, br, bl):

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl

    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)

    M = cv2.getPerspectiveTransform(rect, h)
    warped = cv2.warpPerspective(image, M, ( 450, 450))

    return warped


sudoku_image = cv2.imread ('images/sudoku.jpg')





sudoku_image = shrink_image( sudoku_image )

gray_sudoku = cv2.cvtColor ( sudoku_image, cv2.COLOR_BGR2GRAY)

blur_gray_sudoku = cv2.GaussianBlur( gray_sudoku, (5,5), 0 )
thres_sudoku = cv2.adaptiveThreshold( blur_gray_sudoku,255,1,1,11,2)






contours, hierarchy = cv2.findContours( thres_sudoku,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#print ( len ( contours ) )

#cv2.drawContours(sudoku_image, contours, -1, (0,255,0), 3)

Required_Square_Contour = None
Required_Contour_Area = 0
for Contour in contours:
    Contour_Area = cv2.contourArea(Contour)

    if Contour_Area > 500:
        if Contour_Area > Required_Contour_Area:
            Required_Contour_Area = Contour_Area
            Required_Square_Contour = Contour

Perimeter_of_Contour = cv2.arcLength(Required_Square_Contour, True)
Temp_Square_Countour = cv2.approxPolyDP(Required_Square_Contour, 0.05 * Perimeter_of_Contour, True)
cv2.drawContours(sudoku_image, Required_Square_Contour, -1, (0, 255, 0), 3)







#print( len(Required_Square_Contour) )

#left_least = 1000
#left_top_least = 1000
left_top_corner = [ 1000, 1000]
right_top_corner = [ 0, 1000]
#right_bottom_corner_sum = 0
right_bottom_corner = [ 0,0]
left_bottom_corner = [ 1000, 0]

for i in Required_Square_Contour:

    if ( sum(i[0]) < sum(left_top_corner) ):
        left_top_corner = i[0]

    if ( sum(i[0]) > sum( right_bottom_corner)):
        right_bottom_corner = i[0]

    if ( (i[0][1]-i[0][0]) < (right_top_corner[1]-right_top_corner[0])):
        right_top_corner = i[0]

    if ( (i[0][1]-i[0][0]) > (left_bottom_corner[1]-left_bottom_corner[0])):
        left_bottom_corner = i[0]

    '''
    if ( left_least > i[0][0] and left_top_least > i[0][1]):
        left_top_corner = i[0]

    if ( right_bottom_corner > i[0][0] and left_top_least > i[0][1]):
        right_bottom_corner = i[0]
    '''


#cv2.circle( sudoku_image, (left_top_corner[0], left_top_corner[1]), 15, ( 0, 0, 255), -1)
#cv2.circle( sudoku_image, (right_bottom_corner[0], right_bottom_corner[1]), 15, ( 0, 0, 255), -1)
#cv2.circle( sudoku_image, (right_top_corner[0], right_top_corner[1]), 15, ( 0, 0, 255), -1)
#cv2.circle( sudoku_image, (left_bottom_corner[0], left_bottom_corner[1]), 15, ( 0, 0, 255), -1)
#cv2.rectangle( sudoku_image, (left_top_corner[0], left_top_corner[1]), (right_bottom_corner[0], right_bottom_corner[1]), ( 0, 0, 255), 2)

wrapped_sudoku = wrapper ( sudoku_image, left_top_corner, right_top_corner,
                           right_bottom_corner, left_bottom_corner )

print ( wrapped_sudoku.shape )

#print(len(target))

#wrapped = four_point_transform(thres_sudoku.copy(), target)
#cv2.drawContours(thres_sudoku, [target], -1, ( 0, 255, 0), 3)

cv2.imshow( 'Gray',wrapped_sudoku)
cv2.waitKey(0)
cv2.destroyAllWindows()