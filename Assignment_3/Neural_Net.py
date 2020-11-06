'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

# Importing Required Libraries
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

# Cleaning and Prepreocessing of the Dataset
def boxplots(df,att):
    plot = df.boxplot(column=att)
    plot.set_title("Box Plot for "+att)
    plt.show()

def dataCleaning(df):
    # print("Check for Missing Values:")
    # print(df.count())
    # for i in df:
    #     boxplots(df,i)

    # Median replacement done for the attributes with outliers
    outlier_categories = ['Age','BP','HB']

    for i in outlier_categories:
        df[i] = df[i].fillna(np.nanmedian(df[i]))

    # Mean replacement for other numeric attributes
    df['Weight'] = df['Weight'].fillna(int(np.nanmean(df['Weight'])))

    # Mode replacement done for all binary attributes
    att = ['Delivery phase','Education','Residence']
    for i in att:
        df[i] = df[i].fillna(mode(df[i]))

    # print("Check after dealing with missing values:")
    # print(df.count())

    return df

class NN:

	# Defining some common activation functions

	def sigmoid(self,value):
		return 1/(1 + np.exp(-value))

	def ReLu(self, value):
		return max(0, value)

	def softmax(self, X):
		return np.exp(X)/np.sum(np.exp(X))



	''' X and Y are dataframes '''
	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

	def CM(self,y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")

#path = "D:\\PESU\\Sem 5\\Machine Intelligence\\MI_Assignment\\Assignment_3\\"
#dataset = pd.read_csv(path + os.listdir(path)[-2])
dataset = pd.read_csv("LBW_Dataset.csv")
dataset = dataCleaning(dataset)

	


