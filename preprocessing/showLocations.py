
import numpy as np
import pandas as pd
import cv2

ROOTDIR = '../'
image_dir = ROOTDIR + '/data/2015/'
inputfile = ROOTDIR  + '/data/2015-Z-ALL-COUNTERS.csv'
counts = pd.read_csv(inputfile)

allfile = ROOTDIR  + '/data/2015-Z-LOCATIONS.csv'
allfile = ROOTDIR  + '/data/2015-FINAL-FINAL.csv'
all_counts = pd.read_csv(allfile)

movieList = np.genfromtxt(ROOTDIR + '/data/2015-checked-train.txt',dtype='str')
#np.genfromtxt('col.txt',dtype='str')
endLoop=False
for imagename in movieList: 

    imagename = "SWC1077"
    print(imagename + '.JPG')
    im = cv2.imread(image_dir + imagename + '.JPG')
    rawIm = im.copy()
    allIm = im.copy()
    df = all_counts[all_counts['image_name']==imagename]

    for i,point in df.iterrows():
        allIm = cv2.drawMarker(allIm, (int(point['xcoord']),int(point['ycoord'])), (0,0,255), markerType=cv2.MARKER_SQUARE, markerSize=60, thickness=2, line_type=cv2.LINE_AA)


    imCounts = counts[counts['SWC_image']==imagename]

    for user in imCounts['user_name'].drop_duplicates():
        df = imCounts[imCounts['user_name']==user]
        for i,point in df.iterrows():
            im = cv2.drawMarker(im, (int(point['xcoord']),int(point['ycoord'])), (0,0,255), markerType=cv2.MARKER_STAR, markerSize=30, thickness=2, line_type=cv2.LINE_AA)
        
    cv2.namedWindow(imagename, flags =  cv2.WINDOW_GUI_EXPANDED )
    cv2.imshow(imagename,im)
    cv2.imwrite(imagename + '.png', allIm)
    break
    while(1):
        k = cv2.waitKey(0) & 0xFF
        if k==27:    # Esc key to stop
            endLoop = True
            break
        elif k==ord('c'):
            break
        elif k==ord('r'):
            cv2.imshow(imagename,rawIm)
        elif k==ord('a'):
            cv2.imshow(imagename,allIm)
        elif k==ord('z'):
            cv2.imshow(imagename,im)

    cv2.destroyAllWindows()
    if endLoop:
        break



