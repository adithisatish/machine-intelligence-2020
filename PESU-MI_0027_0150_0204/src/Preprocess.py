import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
from statistics import mode

pd.options.mode.chained_assignment = None # To ignore the SettingWithCopy warning

# Cleaning and Prepreocessing of the Dataset
def boxplots(df,att):
    plot = df.boxplot(column=att)
    plot.set_title("Box Plot for "+att)
    plt.show()

def data_cleaning(df):
	# print("Check for Missing Values:")
    # print(df.count())
    # for i in df:
    #     boxplots(df,i)

	# df = df.drop(columns=['Education', 'Residence'])

    # Median replacement done for the attributes with outliers
	outlier_categories = ['Age','BP','HB']	
	for i in outlier_categories:
		df[i] = df[i].fillna(int(np.nanmedian(df[i])))

		# Clamping the outliers to their boxplot upper and lower bounds
		q1,q3 = np.percentile(df[i],[25,75])
		IQR = q3-q1

		min_clamp = q1-1.5*IQR
		max_clamp = q3+1.5*IQR

		df[i].loc[df[i] > max_clamp] = max_clamp
		df[i].loc[df[i] < min_clamp] = min_clamp

		# Mean replacement of outliers : Giving worse metrics
		# df[i].loc[df[i] > max_clamp] = np.mean(df[i]) 
		# df[i].loc[df[i] < min_clamp] = np.mean(df[i])

	# Mean replacement for other numeric attributes
	df['Weight'] = df['Weight'].fillna(int(np.nanmean(df['Weight'])))

	# Mode replacement done for all other attributes with missing values
	att = ['Delivery phase', 'Education', 'Residence']

	for i in att:
		df[i] = df[i].fillna(mode(df[i]))

	return df

def standardization(df):
	for i in df.keys():
		df[i] = df[i].apply(lambda x: (x-np.mean(df[i])))

	return df

def normalization(df):
	for i in df.keys():
		df[i] = df[i].apply(lambda x: (x-np.mean(df[i])/np.std(df[i])))

	return df

def feature_scaling(df):
	for i in df.keys():
		min_val = min(df[i])
		# max_val = max(df[i])
		range_val = max(df[i]) - min_val

		if range_val==0:
			df[i] = df[i].apply(lambda x: x/min_val)
		else:
			df[i] = df[i].apply(lambda x: (x-min_val)/(range_val))
	return df

def preprocessing(dataset):
	comm = pd.get_dummies(dataset.Community, prefix = "comm")
	# dphase = pd.get_dummies(dataset['Delivery phase'], prefix="dphase")
	residence = pd.get_dummies(dataset.Residence, prefix='res')
	
	dataset = dataset.drop(columns=['Community','Education', 'Delivery phase','Residence'])
	dataset = feature_scaling(dataset)
	
	dataset = dataset.join(comm)
	# dataset = dataset.join(dphase)
	dataset = dataset.join(residence)

	return dataset


path = "\\".join(os.getcwd().split("\\")[:-1] + ['data'])
# print(current_path)
df = pd.read_csv(path + "\\LBW_Dataset.csv")
df = preprocessing(data_cleaning(df))

df.to_csv(path + "\\preprocessed_dataset.csv", index=False)