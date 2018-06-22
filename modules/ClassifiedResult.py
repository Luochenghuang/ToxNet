import pandas as pd
import numpy as np
import os

# add path to the file
archdir = "../result/CNN/"

#input filename you need
filename = "predictions_tox_niehs_epa.csv"

df = pd.read_csv(archdir+filename, header=None)
classes = []
for index, i in df.iterrows():
    classes.append((np.where(i==i.max())[0]+1)[0])

# add to original dataframe
df['classes'] = classes

df.to_csv(archdir+"classified_"+filename, sep='\t')
