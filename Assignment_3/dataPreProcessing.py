import pandas as pd
import numpy as np
import os
from statistics import mode
import matplotlib.pyplot as plt

df = pd.read_csv("LBW_Dataset.csv")

def boxplots(att):
    plot = df.boxplot(column=att)
    plot.set_title("Box Plot for "+att)
    plt.show()

for i in df:
    boxplots(i)

# Median Replacement done for all numeric attributes having outliers

outlier_categories = ['Age','BP','HB']

for i in outlier_categories:
    df[i] = df[i].fillna(np.nanmedian(df[i]))

# Mean Replacement done for other numeric attributes

df['Weight'] = df['Weight'].fillna(int(np.nanmean(df['Weight'])))

# Mode replacement for the binary attributes

att = ['Delivery phase','Education','Residence']
for i in att:
    df[i] = df[i].fillna(mode(df[i]))
# df.count()

# Converting to new dataset

df.to_csv('LBW_Data_Clean.csv')

