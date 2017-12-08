
import numpy as np
import pandas as pd
import cv2

ROOTDIR = '../'
image_dir = ROOTDIR + '/data/2015/'
inputfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
counts = pd.read_csv(inputfile)

movieList = np.genfromtxt(ROOTDIR + '/data/2015-train.txt',dtype='str')
#np.genfromtxt('col.txt',dtype='str')
endLoop=False
showCounts=True
for imagename in movieList: 
    print(imagename + '.JPG')
    im = cv2.imread(image_dir + imagename + '.JPG')
    rawIm = im.copy()
    df = counts[counts['image_name']==imagename]

    for i,point in df.iterrows():
        im = cv2.drawMarker(im, (int(point['xcoord']),int(point['ycoord'])), (0,0,255), markerType=cv2.MARKER_STAR, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
    
    cv2.namedWindow(imagename, flags =  cv2.WINDOW_GUI_EXPANDED )
    cv2.imshow(imagename,im)
    while(1):
        k = cv2.waitKey(0) & 0xFF
        if k==27:    # Esc key to stop
            endLoop = True
            break
        elif k==ord('c'):
            break
        elif k==ord('t'):
            if showCounts:
                cv2.imshow(imagename,im)
                showCounts=False
            else:
                cv2.imshow(imagename,rawIm)
                showCounts=True

    cv2.destroyAllWindows()
    if endLoop:
        break



