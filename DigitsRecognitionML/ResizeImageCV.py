

import cv2

imageBuffer = cv2.LoadImage( 'images/digits_sudoku2.png' )
nW = 468
nH = 99
smallerImage = cv2.CreateImage( (nH, nW), imageBuffer.depth, imageBuffer.nChannels )
cv2.Resize( imageBuffer, smallerImage , interpolation=cv2.CV_INTER_CUBIC )
cv2.SaveImage( 'images/digits_sudoku3.png', smallerImage )