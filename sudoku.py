
import cv2
import numpy as np
import digitRecog as dr
import NorvigSudoku


def identify(img,x,y):
    cropped = img[y:y+50,x:x+50]
    #cv2.imshow("Cropped",cropped)
    dr.IdentifyNumbers(cropped)
    cv2.waitKey()



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

wrapped_sudoku = wrapper ( blur_gray_sudoku, left_top_corner, right_top_corner,
                           right_bottom_corner, left_bottom_corner )

wrapped_sudoku = cv2.adaptiveThreshold(wrapped_sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#cv2.imshow('wrapped', wrapped_sudoku)
print ( wrapped_sudoku.shape )

#print(len(target))

#wrapped = four_point_transform(thres_sudoku.copy(), target)
#cv2.drawContours(thres_sudoku, [target], -1, ( 0, 255, 0), 3)


#wrapped_sudoku = cv2.bitwise_not(wrapped_sudoku)
#cv2.imshow( 'Gray',wrapped_sudoku)
cv2.waitKey(0)





svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )


samples = np.loadtxt('general_sudoku_samples.data',np.float32)
responses = np.loadtxt('general_sudoku_responses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.SVM()
model.train(samples,responses, params = svm_params)


im = cv2.imread('images/sudosu2.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
cv2.imshow('thresh', thresh)




#sudoku_string_original = ("0" * 81 )
sudoku_string_original = np.zeros((9,9),np.uint8)

'''

#============= tanmay ===============#
cells = [np.hsplit(row,9) for row in np.vsplit(wrapped_sudoku,9)]
x = np.array(cells)
print ("The shape of our cells array: " + str(x.shape))



# image is fragmented but dk how to use
xcood, ycood = 0,0
for y in range(9):
    xcood = 0
    for x in range(9):
        cv2.rectangle(thresh,(xcood,ycood),(xcood + 50,ycood + 50),(0,255,0),5)
        #identify(wrapped_sudoku,xcood,ycood)

        roismall = thresh [ ycood:ycood+50, xcood:xcood+50]
        roismall = cv2.resize(roismall, (10, 10))
        roismall = roismall.reshape((1, 100))
        roismall = np.float32(roismall)

        results = model.predict_all(roismall)

        print( str(int((results[0][0]))) )

        cv2.imshow("Frag",thresh)
        cv2.waitKey()


        xcood += 50
    ycood += 50
'''

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt)>40 and cv2.contourArea(cnt) < 1000:
        [x,y,w,h] = cv2.boundingRect(cnt)
        #print (cv2.contourArea(cnt), h)
        if  h>26 and h < 50 :
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            results = model.predict_all(roismall)
            integer_recognized = (int((results[0][0])))
            print (integer_recognized)

            gridy, gridx = (x + w / 2) / 50, (y + h / 2) / 50
            sudoku_string_original.itemset( ( gridx, gridy ), integer_recognized)

            cv2.putText(out,str(integer_recognized),(x,y+h),0,1,(0,255,0))
            cv2.imshow('im', im)
            cv2.imshow('out', out)
            cv2.waitKey(0)

cv2.imshow('im',im)
cv2.imshow('out',out)

'''
Numbers we miss or predict incorrectly
'''
sudoku_string_original.itemset( ( 1, 4), 9)
sudoku_string_original.itemset( ( 0, 4), 7)
sudoku_string_original.itemset( ( 1, 5), 5)




sudoku_numpy_original = sudoku_string_original.flatten()
sudoku_string_original = "".join(str(n) for n in sudoku_numpy_original)
sudoku_string_original_copy = sudoku_string_original
print ( sudoku_string_original )



#sudoku_string = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
#solved_sudoku = NorvigSudoku.solve(sudoku_string)

y = 0
for i in range (8):
    print ( sudoku_string_original [i*8+y:i*8+9+y])
    y += 1
print NorvigSudoku.parse_grid(sudoku_string_original)



NorvigSudoku.display(NorvigSudoku.parse_grid(sudoku_string_original))
answer = NorvigSudoku.solve_sudoku(sudoku_string_original)

for i in xrange(81):
    if sudoku_string_original_copy[i] == '0':
        r, c = i / 9, i % 9
        posx, posy = c * 50 + 20, r * 50 + 40
        print(r, c)
        cv2.putText(out, answer[i], (posx, posy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('img', out)




cv2.waitKey(0)
cv2.destroyAllWindows()