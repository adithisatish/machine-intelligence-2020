'''
MI Assignment 1

Authors: 
- Adithi Satish - PES1201800150
- Ananya Veeraraghavan - PES1201800204
- Shriya Shankar - PES1201800027
'''

'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):

	''' The entropy of the dataset can be defined as
	
	E(S) = - Summation((pi/n)*log(pi/n)) for all i from 1 to C, where
	n is the total number of instances in the dataset,
	C is the number of categories/class labels of the target variable,
	pi is the number of instances where target variable = class i.
	'''

	entropy = 0
	size = len(df) #the total number of instances in the dataset
	if size == 0:
		return entropy
	
	target = list(df.columns)[-1] #the target variable i.e Nth column in the dataset
	label_count = dict(df[target].value_counts()) #keys in the dictionary are all the unique class labels with the values being their corresponding counts
	
	for i in label_count:
		p = label_count[i] #the number of instances in the dataset where target = i
		res = -(p/size)*np.log2([p/size])[0]
		entropy += res
	
	#print(entropy)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	
	''' The entropy of an attribute, or average information can be defined as 
	I(Attribute) = Sum(Probability(Attribute = i)*Entropy(Attribute = i)) where i represents all the unique values said attribute can take.
	'''

	entropy_of_attribute = 0
	size = len(df) #the total number of instances in the dataset

	if size==0:
		return entropy_of_attribute

	values = list(df[attribute].unique()) #all possible values the given attribute can take

	for i in values:
		new_df = df[df[attribute]==i]
		entropy = get_entropy_of_dataset(new_df) #Entropy(Attribute = i)
		entropy_of_attribute+= len(new_df)/size * entropy #len(new_df)/size represents Probability(Attribute = i)

	#print(entropy_of_attribute)
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):

	'''The information gain of an attribute is defined as the difference between the entropy of the entire dataset and the average information of the given attribute.
	G(S,A) = E(S) - I(A)
	'''
	information_gain = 0

	entropy_df = get_entropy_of_dataset(df) #E(S)
	avg_info = get_entropy_of_attribute(df,attribute) #I(A)
	
	information_gain = entropy_df - avg_info #G(S,A)
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''
	max = 0
	
	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
	if(len(df.columns) < 1): #To take care of the case with no attributes
		return (information_gains,selected_column)

	if(len(df.columns)==1): #To take care of the case with only one attribute in the dataset
		col = list(df.columns)[0]
		information_gains[col] = get_information_gain(df,col)
		selected_column = col

	else:
		columns = list(df.columns)[:-1] #All columns except the target variable

		for i in columns:
			information_gains[i] = get_information_gain(df,i)
			if max < information_gains[i]:
				max = information_gains[i]
				selected_column = i
		#print(information_gains)
		#max_info_gain = max(information_gains.values())
		#selected_column = [i for i in information_gains if information_gains[i]==max_info_gain][0]

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
