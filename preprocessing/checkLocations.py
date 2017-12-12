
import numpy as np
import pandas as pd
import cv2

ROOTDIR = '../'
image_dir = ROOTDIR + '/data/2015/'

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
all_counts = pd.read_csv(allfile)

trainList = np.genfromtxt(ROOTDIR + '/data/2015-train.txt',dtype='str')
testList = np.genfromtxt(ROOTDIR + '/data/2015-test.txt',dtype='str')

count = len(trainList)
swap=0
c=0

endLoop=False
while c<count:
    imagename = trainList[c]
    print(c,imagename)
    im = cv2.imread(image_dir + imagename + '.JPG')
    rawIm = im.copy()
    allIm = im.copy()
    df = all_counts[all_counts['image_name']==imagename]

    for i,point in df.iterrows():
        allIm = cv2.drawMarker(allIm, (int(point['xcoord']),int(point['ycoord'])), (0,0,255), markerType=cv2.MARKER_SQUARE, markerSize=60, thickness=2, line_type=cv2.LINE_AA)


        
 #   cv2.namedWindow(imagename,
    cv2.namedWindow(imagename, flags =  cv2.WINDOW_GUI_EXPANDED) # + cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow(imagename, 1800,1800)
    #cv2.setWindowProperty(imagename,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(imagename,allIm)
    while(1):
        k = cv2.waitKey(0) & 0xFF
        if k==27:    # Esc key to stop
            endLoop = True
            break
        elif k==ord('y'):
            c=c+1
            break
        elif k==ord('n'):
            trainList[c] = testList[swap]
            testList[swap] = imagename
            swap = swap+1
            break


    cv2.destroyAllWindows()
    if endLoop:
        break



np.savetxt('../data/2015-checked-train.txt', trainList,fmt='%s') 
np.savetxt('../data/2015-checked-test.txt', testList,fmt='%s') 
