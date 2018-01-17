
import cv2
import numpy as np
import digitRecog as dr


def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape( (4,2) )
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped



sudoku_image = cv2.imread ('images/sudosu.jpg')
gray_sudoku = cv2.cvtColor ( sudoku_image, cv2.COLOR_BGR2GRAY)

blur_gray_sudoku = cv2.GaussianBlur( gray_sudoku, (5,5), 0 )
thres_sudoku = cv2.adaptiveThreshold( blur_gray_sudoku,255,1,1,11,2)

contours, hierarchy = cv2.findContours( thres_sudoku,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print ( len ( contours ) )

print ( contours[0])
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


for c in contours:
    
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        cv2.drawContours(thres_sudoku, c, -1, (0, 255, 0), 3)
        break



#print(len(target))

#wrapped = four_point_transform(thres_sudoku.copy(), target)
#cv2.drawContours(thres_sudoku, [target], -1, ( 0, 255, 0), 3)

cv2.imshow( 'Gray',sudoku_image)
cv2.waitKey(0)
cv2.destroyAllWindows()