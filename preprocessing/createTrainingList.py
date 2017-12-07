import numpy as np
import os
import pandas as pd

np.random.seed(42)
left = lambda x: x[:7]

image_names = [f for f in os.listdir('../data/2015/') if os.path.isfile(os.path.join('../data/2015', f))]
image_names = pd.DataFrame(image_names)
image_names = image_names.drop_duplicates()[0].apply(left)

print(image_names.shape[0])
image_names = image_names.iloc[np.random.permutation(image_names.shape[0])]

train = image_names[:500]
test = image_names[500:1500]

train.to_csv('../data/2015-train.txt', header=None, index=None, sep=' ', mode='a')
test.to_csv('../data/2015-test.txt', header=None, index=None, sep=' ', mode='a')
