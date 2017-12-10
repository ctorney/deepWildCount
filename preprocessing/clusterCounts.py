import numpy as np
import pandas as pd
import sklearn.cluster as skc
import scipy.sparse as ss
from math import hypot


def allocate2Cluster(xmarks, ymarks, xcluster, ycluster):
    nm = len(xmarks)
    nc = len(xcluster)
    returnVals = -1*np.ones(nm)
    distances = np.full((nm,nc), np.inf)

    # calculate distances between marks and clusters
    for i in range(nm):
        for j in range(nc):
            distances[i][j] = hypot(xmarks[i]-xcluster[j],ymarks[i]-ycluster[j])
    # any marks over 64 pixels cannot belong to that cluster
    distances[distances>64]=np.inf
    
    # loop while there are still potential matchings
    while np.any(np.isfinite(distances)): 
        # find the closest cluster-mark pairing
        [i,j] = np.argwhere(distances == np.min(distances))[0]
        # match the two closest
        returnVals[i]=j
        # set that cluster to infinity for all other marks 
        distances[:,j]=np.inf

    # anything left over now has a new cluster created
    next_c=nc
    for i in range(nm):
        if returnVals[i]<0:
            returnVals[i]=next_c
            next_c+=1

    return returnVals



ROOTDIR = '../'
image_dir = ROOTDIR + '/data/2015/'
inputfile = ROOTDIR  + '/data/2015-Z-ALL-COUNTERS.csv'
count_data = pd.read_csv(inputfile)
count_data.drop_duplicates(inplace=True)

data = []


# uses clustering to group identifications by multiple counters into one average location
for image in count_data['SWC_image'].drop_duplicates():
    df = count_data.loc[count_data['SWC_image'] == image][['user_name', 'xcoord', 'ycoord']].dropna(axis=0)
    df['cluster']=-1
    for user, marks in df.groupby('user_name'):
        cluster_locations = df[df['cluster']>=0].groupby(['cluster']).mean()
        allocated = allocate2Cluster( marks.xcoord.values,marks.ycoord.values, cluster_locations.xcoord.values, cluster_locations.ycoord.values)
        j=0
        for i,m in marks.iterrows():
            df.cluster[i]=allocated[j]
            j+=1
    c_count = df['cluster'].max()+1

    for cluster in range(c_count):
        if len(df[df.cluster==cluster]) > 5:
            Xx = df[df.cluster==cluster].xcoord.mean()
            Yy = df[df.cluster==cluster].ycoord.mean()
            data.append([image,Xx,Yy])



Y = pd.DataFrame(data, columns=['image_name','xcoord','ycoord'])
Y.to_csv('../data/2015-FINAL-FINAL.csv')
