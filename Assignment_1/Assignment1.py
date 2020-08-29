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
	entropy = 0
	size = len(df) #the total number of instances in the dataset
	target = list(df.columns)[-1] #the target variable i.e Nth column in the dataset
	values = list(df[target].unique()) #all possible values the target variable can take

	for i in values:
		p = len(df[df[target]==i]) #the number of instances in the dataset where target = i
		if p==0:
			res = 0
		else:
			res = -(p/size)*np.log2([p/size])[0]
		entropy+=res
	
	#print(entropy)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	size = len(df) #the total number of instances in the dataset
	target = list(df.columns)[-1] #the target variable i.e Nth column in the dataset
	values = list(df[attribute].unique()) #all possible values the given attribute can take

	for i in values:
		new_df = df[df[attribute]==i]
		entropy = get_entropy_of_dataset(new_df)
		entropy_of_attribute+= len(new_df)/size * entropy

	#print(entropy_of_attribute)
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0

	entropy_df = get_entropy_of_dataset(df)
	avg_info = get_entropy_of_attribute(df,attribute)
	
	information_gain = entropy_df - avg_info
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''
	
	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	columns = list(df.columns)[:-1] #All columns except the target variable

	for i in columns:
		information_gains[i] = get_information_gain(df,i)
	
	max_info_gain = max(information_gains.values())
	selected_column = [i for i in information_gains.keys() if information_gains[i]==max_info_gain][0]

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''