
import cv2
import numpy as np

ROI_X_Width = 25
ROI_Y_Height = 35
###The size of the training sample
Training_Image = cv2.imread('trainio.png')
Training_Output = Training_Image.copy()
###Read the traing image
Modified_Training_Image = cv2.cvtColor(Training_Image,cv2.COLOR_BGR2GRAY)
Modified_Training_Image = cv2.GaussianBlur(Modified_Training_Image,(5,5),0)
Modified_Training_Image = cv2.adaptiveThreshold(Modified_Training_Image,255,1,1,11,2)
Training_Output = cv2.bitwise_not(Training_Output)
###Reduce noise and then convert it to binary image
Contours, Hierarchy = cv2.findContours(Modified_Training_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
###Find Contours
Samples =  np.empty((0,ROI_X_Width*ROI_Y_Height))
###Creates an empty array of Samples which will be used to store the pixel values
for Contour in Contours:
    if cv2.contourArea(Contour)>10:
        [Abscissa,Ordinate,X_Width,Y_Height] = cv2.boundingRect(Contour)
        ###Selection criteria for preventing dots(noise)
        if  Y_Height>18:
            ###Another selection criteria (not really required)
            Region_of_Interest = Modified_Training_Image[Ordinate:Ordinate+Y_Height,Abscissa:Abscissa+X_Width]
            Region_of_Interest = cv2.resize(Region_of_Interest,(ROI_X_Width,ROI_Y_Height))
            ###Selects the digits one by one and draws a fitting rectangle around it and waits for a manual keypress
            Sample = Region_of_Interest.reshape((1,ROI_X_Width*ROI_Y_Height))
            Samples = np.append(Samples,Sample,0)
            ###Saves the corresponding pixel values in Samples
print "Training Complete"
np.savetxt('Samples.data',Samples)

ROI_X_Width = 25
ROI_Y_Height = 35
###The size of the training sample
### Loading the trained data ###
Samples = np.loadtxt('Samples.data',np.float32)
Responses = np.loadtxt('Responses.data',np.float32)
Responses = Responses.reshape((Responses.size,1))
### Train using the data ###
model = cv2.KNearest()
model.train(Samples,Responses)
###We use a model variable because there can be multiple models to train on in the same code
Warped_Sudoku_Image = cv2.imread('../' + sudoku_image_path)
###Load the image to process
Output_Image = np.zeros(Warped_Sudoku_Image.shape,np.uint8)
###Create an black image of same size
Modified_Sudoku_Image = cv2.cvtColor(Warped_Sudoku_Image,cv2.COLOR_BGR2GRAY)
Modified_Sudoku_Image = cv2.adaptiveThreshold(Modified_Sudoku_Image,255,1,1,11,2)
###Reduce noise
Contours,Hierarchy = cv2.findContours(Modified_Sudoku_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
###Find Contours
Sudoku_Text = [[0] * 9 for i in range(9)]
###Create a 9x9 list of all zeros i.e. a blank sudoku
### Recognition part ###
for Contour in Contours:
    ###Select a contour one by one
    if cv2.contourArea(Contour)>30 and cv2.contourArea(Contour)<1000:
        ###Selection criteria for preventing selection of dots(noise) and sudoku grid squares
        [Abscissa,Ordinate,X_Width,Y_Height] = cv2.boundingRect(Contour)
        if  Y_Height>22 and Y_Height<40:
            ###Selection criteria for preventing selection of sudoku grid
            cv2.rectangle(Warped_Sudoku_Image,(Abscissa,Ordinate),(Abscissa+X_Width,Ordinate+Y_Height),(0,255,0),2)
            ###Selects the digits one by one and draws a fitting rectangle around it
            Region_of_Interest = Modified_Sudoku_Image[Ordinate:Ordinate+Y_Height,Abscissa:Abscissa+X_Width]
            ###Selects the number in question
            Region_of_Interest = cv2.resize(Region_of_Interest,(ROI_X_Width,ROI_Y_Height))
            ###Resizes it to a 10x10 image
            Region_of_Interest = Region_of_Interest.reshape((1,ROI_X_Width*ROI_Y_Height))
            ###Converts the 10x10 image into an array of 100 pixel values
            Region_of_Interest = np.float32(Region_of_Interest)
            ###Converts it into float32 type
            Retval, Result, Neigh_Resp, Dists = model.find_nearest(Region_of_Interest, k = 1)
            ###Apply kNN algorithm to find nearest neighbours
            string = str(int((Result[0][0])))
            ###Converts the result into an integer and then a string to put on the output image
            cv2.putText(Output_Image,string,(Abscissa,Ordinate+Y_Height),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
            ###cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            ###(img) ~Image.
            ###(text) ~Text string to be drawn.
            ###(org) ~Bottom-left corner of the text string in the image.
            ###(font) ~CvFont structure initialized using InitFont().
            ###(fontFace) ~Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
            ## FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX,
            ## or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID's can be combined with FONT_ITALIC to get
            ## the slanted letters
            ###(fontScale) ~Font scale factor that is multiplied by the font-specific base size.
            ###(color) ~Text color
            Sudoku_Text[(Ordinate+Y_Height)/50][Abscissa/50] = int(string)
            ### row=(Ordinate+Y_Height)/50 since it is a 450x450 grid, same for col
            ## Hence Sudoku_Text stores the values of the identified difit
            ## in its respective place
#print np.asarray(Sudoku_Text)
#cv2.imshow('Warped_Sudoku_Image',Warped_Sudoku_Image)
cv2.imwrite('../images/sudoku_with_boundary',Warped_Sudoku_Image)
#cv2.imshow('Output_Image',Output_Image)
cv2.imwrite('../images/recognitions',Output_Image)
#cv2.waitKey(0)
#print Sudoku_Text
return Sudoku_Text
