import numpy as np
import pandas as pd
import sklearn.cluster as skc
import scipy.sparse as ss


# uses clustering to group identifications by multiple counters into one average location

data_file_csv = '../data/swc_zooniverse_data_22Nov17.csv'
count_data = pd.read_csv(data_file_csv)
data = []

for tile in count_data['tile_id'].drop_duplicates():
    df = count_data.loc[count_data['tile_id'] == tile].dropna().groupby('user_name')['mark_index'].nunique()
    if len(df.index) > 0:
        accepted_number_of_marks = max(df)
    else:
        accepted_number_of_marks = 0
    print('Tile: ' + tile + ', accepted marks: ' + str(accepted_number_of_marks))
    if accepted_number_of_marks >= 2:
        X = count_data.loc[count_data['tile_id'] == tile][['user_name', 'xcoord', 'ycoord']].dropna(axis=0)
        # Blank coo matrix to csr
        sm = ss.coo_matrix((X.shape[0], X.shape[0]), np.float32).tolil()
        # Insert 1 for connected pairs and diagonals
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):# add entries based on username, distance
                if i == j:
                    sm[i,j] = 1.0
                else:
                    if X.iloc[i]['user_name'] == X.iloc[j]['user_name']:
                        sm[i,j] = 0.0005
                    else:
                        if (abs(X.iloc[i]['xcoord'] - X.iloc[j]['xcoord']) >= 32) or (abs(X.iloc[i]['ycoord'] - X.iloc[j]['ycoord']) >= 32):
                            sm[i,j] = 0.0005
                        else:
                            sm[i,j] = 1.0
        sm = sm.tocoo()  # convert back to coo format
        model = skc.AgglomerativeClustering(accepted_number_of_marks, connectivity=sm, linkage='ward')
        y = model.fit_predict(X[['xcoord', 'ycoord']])
        X['cluster'] = y
        X = X.groupby('cluster').filter(lambda x: len(x) >= 5)
        X = X.groupby('cluster').mean()
        X['tile_id'] = tile
        for index, row in X.iterrows():
            data.append([row.tile_id,index,row.xcoord,row.ycoord])
    elif accepted_number_of_marks == 1:
        X = count_data.loc[count_data['tile_id'] == tile].dropna(axis = 0)
        if len(X) > 5:
            Xx = sum(X['xcoord']) / X.shape[0]
            Xy = sum(X['ycoord']) / X.shape[0]
            data.append([tile,0,Xx,Xy])

Y = pd.DataFrame(data, columns=['tile_id','cluster','xcoord','ycoord'])
Y.to_csv('../data/swc_zooniverse_cluster_found_coords.csv')
