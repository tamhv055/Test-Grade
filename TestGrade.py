from imutils.perspective import four_point_transform
from imutils.perspective import order_points
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2



ANSWER_KEY = {0 : 1 , 1 : 1 , 2 : 1 , 3 : 0 , 4 : 1 , 5 : 1 , 6 : 1 , 7 : 1 , 8 : 1 , 9 : 1 , 10 : 1 , 11 : 1 , 12 : 1 , 13 : 1 , 14 : 1 , 15 : 1 , 16 : 1 , 17 : 1 , 18 : 1 , 19 : 1 , 20 : 1 , 21 : 1 , 22 : 1 , 23 : 1 , 24 : 1 , 25 : 2 , 26 : 2 , 27 : 2 , 28 : 2 , 29 : 2 , 30 : 2 , 31 : 2 , 32 : 2 , 33 : 2 , 34 : 2 , 35 : 2 , 36 : 2 , 37 : 2 , 38 : 2 , 39 : 2 , 40 : 2 , 41 : 2 , 42 : 2 , 43 : 2 , 44 : 2 , 45 : 2 , 46 : 2 , 47 : 2 , 48 : 2 , 49 : 2 , 50 : 3 , 51 : 3 , 52 : 3 , 53 : 3 , 54 : 3 , 55 : 3 , 56 : 3 , 57 : 3 , 58 : 3 , 59 : 3 , 60 : 3 , 61 : 3 , 62 : 3 , 63 : 3 , 64 : 3 , 65 : 3 , 66 : 3 , 67 : 3 , 68 : 3 , 69 : 3 , 70 : 3 , 71 : 3 , 72 : 3 , 73 : 3 , 74 : 2 , 75 : 0, 76 : 0 , 77 : 0 , 78 : 0 , 79 : 0 , 80 : 0 , 81 : 0 , 82 : 0 , 83 : 0 , 84 : 0 , 85 : 0 , 86 : 0 , 87 : 0 , 88 : 0 , 89 : 0 , 90 : 0 , 91 : 0 , 92 : 0 , 93 : 0 , 94 : 0 , 95 : 0 , 96 : 0 , 97 : 0 , 98 : 0 , 99 : 0}
resultfile = 'c:\\Users\\tamhv\\Desktop\\res.jpg'
dirfile= 'c:\\Users\\tamhv\\Desktop\\imageTest\\imageConvert\\'
nameImage = 'img0' #-25degree
color=(0,255,0)
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged
x=1   


############ tien xu ly anh

acf=x
image = cv2.imread("c:\\Users\\tamhv\\Desktop\\img02-25degree.jpg",1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged =  auto_canny(gray)
#blurred = cv2.GaussianBlur(edged, (5, 5), 0)

'''kernel = np.ones((1, 1), np.uint8)
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
                             
output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
                 

opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
dilation = cv2.dilate(closing,kernel,iterations = 3)

#edged =cv2.filter2D(dilation, -1, kernel_sharpen_1)
'''



################## xac dinh 4 goc cua to giay
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
            
#color=(0,0,255)
#cv2.drawContours(image,[docCnt],-1, color, 30)

##############
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))	
paper = cv2.resize(paper,(int(990),int(1280)))
warped = cv2.resize(warped,(int(990),int(1280)))

#blurred = cv2.GaussianBlur(warped, (5, 5), 0)		
#sharpen = cv2.filter2D(blurred, -1, kernel_sharpen_3)
#Histeq = cv2.equalizeHist (warped)
#edged =  cv2.Canny(warped, 75, 200)


thresh =cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

#################### xac dinh 3 khu vuc : khu vuc to ID sinh vien , khu vuc to ma de thi , khu vuc to trac nghiem

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


areaArray = []
for i, c in enumerate(cnts):
    area = cv2.contourArea(c)
    areaArray.append(area)

#first sort the array by area
sorteddata = sorted(zip(areaArray, cnts), key=lambda x: x[0], reverse=True)

questionCnts = []
infoCnt =[]

#find the nth largest contour [n-1][1], in this case 2

for i in range(len(sorteddata)) :
    x , y , w , h = cv2.boundingRect(sorteddata[i][1])
    if y > 490:
        questionCnts.append(sorteddata[i][1])
    if y < 490 and x > 605:
        infoCnt.append(sorteddata[i][1])
        

############## xac dinh vung chua ma de thi va mssv		
     
newCnt =[]		
if len(infoCnt) >= 2:
    for c in infoCnt:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area >= 36000:
            newCnt.append(approx)

infoCnt= newCnt


        
    

##### xac dinh thong tin ma SV va ma de thi
if len(infoCnt) == 2 :
    infoCnt = contours.sort_contours(infoCnt,method="left-to-right")[0]

color=(0,255,0)
cv2.drawContours(paper, infoCnt, -1, color, 3)
##### xac dinh so bao danh cua hoc sinh


IDnumber=''

x,y,w,h =cv2.boundingRect(infoCnt[0])

cropRec=paper[y+5: y + h-5, x+5: x + w-5]
cropgray = cv2.cvtColor(cropRec, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(cropgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,127, 0)
### loai bo soc ngang va soc doc  
horizontal = thresh
vertical = thresh
rows,cols = horizontal.shape


horizontalsize = cols /5

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

horizontal= cv2.bitwise_not(horizontal)

verticalsize = rows / 5
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
vertical = cv2.bitwise_not(vertical)

masked_img = cv2.bitwise_and(thresh, thresh, mask=horizontal)
masked_img = cv2.bitwise_and(masked_img, masked_img, mask=vertical)




#### quet so bao danh cua sinh vien: IDnumber
cnts = cv2.findContours(masked_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
choiceCnts=[]
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 10 and h >= 10 and ar >= 0.8 and ar <= 1.2:
        choiceCnts.append(c)

choiceCnts = contours.sort_contours(choiceCnts,method="left-to-right")[0]

for (q, i) in enumerate(np.arange(0, len(choiceCnts), 10)):
    cnts = contours.sort_contours(choiceCnts[i: i+10])[0]
    cnts = contours.sort_contours(cnts,method="top-to-bottom")[0]
    bubbled = None
    countCheck=0
    
    for (j, c) in enumerate(cnts):
    
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        
        #mask4 = cv2.bitwise_and(thresh, thresh, mask=mask)
        #cv2.imshow("mask 4", mask4)
        mask3 = np.zeros(thresh.shape, dtype="uint8")
        cv2.bitwise_and(thresh, thresh, mask3,mask)

        total = cv2.countNonZero(mask3)
        
        if  total >= 165 :
            countCheck+=1
            
        if bubbled is None or total > bubbled[0] :
            bubbled = (total, j)
                
    if countCheck!=1:
        bubbled = (0,50)
                    
        
    if bubbled[1]>=0 and bubbled[1]<=9 :	
        IDnumber=IDnumber+str(bubbled[1])
        cv2.drawContours(cropRec,cnts[bubbled[1]], -1, color, 3)
    else:
        IDnumber=IDnumber+'X'	
   

        
    
print('student ID is: ',IDnumber)



###################### xac dinh ma de thi
testNumber=''
x,y,w,h =cv2.boundingRect(infoCnt[1])

cropRec=paper[y+5: y + h-5, x+5: x + w-5]
cropgray = cv2.cvtColor(cropRec, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(cropgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,127, 0)

###loai bo soc ngang va soc doc
horizontal = thresh
vertical = thresh
rows,cols = horizontal.shape

horizontalsize = cols /5
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
horizontal= cv2.bitwise_not(horizontal)

verticalsize = rows /9
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
vertical = cv2.bitwise_not(vertical)

masked_img = cv2.bitwise_and(thresh, thresh, mask=horizontal)
masked_img = cv2.bitwise_and(masked_img, masked_img, mask=vertical)

##### quet ma de thi testNumber 
cnts = cv2.findContours(masked_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
choiceCnts=[]
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 10 and h >= 10 and ar >= 0.8 and ar <= 1.2:
        choiceCnts.append(c)

choiceCnts = contours.sort_contours(choiceCnts,method="left-to-right")[0]

for (q, i) in enumerate(np.arange(0, len(choiceCnts), 10)):
    cnts = contours.sort_contours(choiceCnts[i: i+10])[0]
    cnts = contours.sort_contours(cnts,method="top-to-bottom")[0]
    bubbled = None
    countCheck=0
    
    for (j, c) in enumerate(cnts):
    
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)			
        total = cv2.countNonZero(mask)
        if  total >= 165 :
            countCheck+=1
            
        if bubbled is None or total > bubbled[0] :
            bubbled = (total, j)
                
    if countCheck!=1:
        bubbled = (0,50)
                    
        
    if bubbled[1]>=0 and bubbled[1]<=9 :	
        testNumber=testNumber+str(bubbled[1])
        cv2.drawContours(cropRec,cnts[bubbled[1]], -1, color, 3)
    else:
        testNumber=testNumber+'X'	
    
print('Test code is :',testNumber)




###################### xac dinh form chua cau tra loi




newCnt=[]
if len(questionCnts) >= 2:
    for c in questionCnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area >= 70000:
            newCnt.append(approx)
questionCnts= newCnt

###### sap xep khung trac nghiem theo thu tu trai sang phai
questionCnts = contours.sort_contours(questionCnts,method="left-to-right")[0]
color=(0,255,0)
cv2.drawContours(paper,questionCnts, -1, color, 3)
###### thuc hien cham bai thi 
correct=0
numberChoice = 4 # co 4 su lua chon
choices= []
count=0
corrects=0
for c in questionCnts:
    choiceCnts = []
    x,y,w,h =cv2.boundingRect(c)
    ### cat tung khung to trac nghiem va cham theo tung khung
    cropRec=paper[y+5: y + h-5, x+5: x + w-5]
    cropgray = cv2.cvtColor(cropRec, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cropgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,127, 0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    ##### xac dinh cac diem khoang tron trong khung to trac nghiem 
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 10 and h >= 10 and ar >= 0.8 and ar <= 1.2:
            choiceCnts.append(c)
            
    ### sap xep cac diem khoang tron thanh tung dong tu tren xuong
    choiceCnts = contours.sort_contours(choiceCnts,method="top-to-bottom")[0]
    
    for (q, i) in enumerate(np.arange(0, len(choiceCnts), numberChoice)):
        ### sap xep thu tu cac diem khoang tron cua 1 cau hoi tu trai sang phai
        cnts = contours.sort_contours(choiceCnts[i:i + numberChoice])[0]
        cnts = contours.sort_contours(cnts,method="left-to-right")[0]
        bubbled = None
        countCheck=0
        for (j, c) in enumerate(cnts):
            
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)	
                        
            total = cv2.countNonZero(mask)
            if  total >= 165 :
                countCheck+=1
        
            if bubbled is None or total > bubbled[0] :
                bubbled = (total, j)
             
        if countCheck!=1:
            bubbled = (0,50)
            
        
        color = (0, 0, 255)
        if (count*25 + q)<100:
            k = ANSWER_KEY[count*25 + q]
        
        
        if k == bubbled[1] :	
            color = (0, 255, 0)
            correct += 1
            
        cv2.drawContours(cropRec, [cnts[k]], -1, color, 3)
    
    
    
    
score=correct
     
#cv2.imwrite('c:\\Users\\tamhv\\Desktop\\result1.jpg', cropRec )
print("[RESULT] score: {:.2f}%".format(score))
cv2.putText(paper,"student ID: "+ str(IDnumber), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.putText(paper, "Test ID: "+ str(testNumber), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.putText(paper, str(score)+'/100', (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


cv2.imwrite("c:\\Users\\tamhv\\Desktop\\result.jpg",paper)      
paper = cv2.resize(paper, (541, 700))
cv2.imshow("paper", paper)


#cv2.imwriteimwrite(dirfile+'resultTestGrade\\result'+nameImage+str(acf)+'-25degree'+'.jpg',paper)

cv2.waitKey(0)
cv2.destroyAllWindows()


































'''
questionCnts = []



for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.2:
        questionCnts.append(c)




questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]

cv2.drawContours(paper, questionCnts, -1, color, 3)			

cv2.imshow("paper", paper)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''correct = 0

cv2.drawContours(newform, questionCnts, -1, color, 3)
cv2.imwrite(dirfile + 'FormCnt.jpg', thresh.copy())


print(len(questionCnts))

for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
    bubbled = None
    
    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # "bubble" for the question
        
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask,[c], -1, 255, -1)
        
        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        total = cv2.countNonZero(mask)
        
        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initialize the contour color and the index of the
    # *correct* answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    print(bubbled[1])
    # check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # draw the outline of the correct answer on the test
    cv2.drawContours(newform, questionCnts, -1, color, 3)



    
    
# grab the test taker
score = (correct / 25.0) * 100

print(score)

cv2.imshow("paper", newform)
#cv2.imshow("warped", warped)
#cv2.imshow("thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(resultfile,image)
'''



#http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html


#edged =  cv2.Canny(warped, 75, 200)





#cv2.imshow("result", thresh)





'''lines = cv2.HoughLinesP(thresh,5,np.pi/180,100,minLineLength=910,maxLineGap=3)



minX1=10000
maxX2=0
minY1=10000
maxY2=0


for line in lines:
    x1,y1,x2,y2 = line[0]
    #cv2.line(paper,(x1,y1),(x2,y2),(0,255,0),7)
    if x1 < minX1 :
        minX1=x1
    if y1 < minY1 :
        minY1=y1	
    if x2 > maxX2:
        maxX2=x2
    if y2 > maxY2:
        maxY2=y2
        

newpaper = paper[minY1:maxY2,minX1:maxX2]
papergray = cv2.cvtColor(newpaper , cv2.COLOR_BGR2GRAY)
cv2.imwrite(dirfile+'resultCut\\' +'Res-'+ nameImage, newpaper)	



thresh = cv2.threshold(papergray, 125, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]



cv2.imshow("paper", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

rectangleCnts = []


if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)		
        if len(approx) == 4:
            rectangleCnts.append(approx)

cv2.drawContours(paper, rectangleCnts, -1, color, 3)			

cv2.imshow("paper", paper)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(len(formCnt) ):
        for j in range (i,len(formCnt) ) :
            if formCnt[i][0][0][0]>formCnt[j][0][0][0]:
                formCnt[i] ,formCnt[j]=formCnt[j],formCnt[i]
        
formCnt[0] = order_points(formCnt[0].reshape(4,2))	

#newform = newpaper[8:772,20:224]
newform = newpaper[int(formCnt[0][0][0]):int(formCnt[0][2][1]),int(formCnt[0][0][1]):int(formCnt[0][1][0])]
#newform = four_point_transform(newpaper, formCnt[0].reshape(4, 2))
warpedform = cv2.cvtColor(newform , cv2.COLOR_BGR2GRAY)
opening = cv2.morphologyEx(warpedform, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
dilation = cv2.dilate(closing,kernel,iterations = 3)

#blurred = cv2.GaussianBlur(dilation, (5, 5), 0)
#output_1 = cv2.filter2D(dilation, -1, kernel_sharpen_1)

thresh = cv2.threshold(dilation, 150, 255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]

#edged =  cv2.Canny(blurred, 75, 150)
thresh = cv2.filter2D(thresh, -1, kernel_sharpen_3)
'''


'''secondlargestcontour = sorteddata[1][1]


x, y, w, h = cv2.boundingRect(secondlargestcontour)
cv2.drawContours(paper, infoCnt, -1, (255, 0, 0), 2)
'''
#cv2.rectangle(paper, (x, y), (x+w, y+h), (0,255,0), 2)

#questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]

'''if len(questionCnts) > 0:
    questionCnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            RecCnt.append(approx)
            
'''