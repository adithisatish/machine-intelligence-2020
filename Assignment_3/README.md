# Designing a Neural Network using NumPy
## Machine Intelligence - Assignment 3

Design of a neural network to implement a binary classifier using the ```LBW_Dataset.csv ``` dataset file.

Functions Implemented:
- ```dataCleaning```: Function to clean the dataset, using mean, median and mode replacement methods.
- ```boxplots```: Function to plot boxplots for an attribute of a dataframe.
- ```NN.fit```: Method of class NN, used to train the neural network using the X_train and y_train parameters.
- ```NN.predict```: Method of class NN, returns the predicted class (y_hat) that a test instance has been classifed as. 
- ```NN.CM```: Method of class NN to print the confusion matirx by taking y_pred and y_test as parameters.

Parameters and Hyperparameters:
- learning rate = 0.05 : a value larger than 0.05 led to divergence whereas a value lesser than 0.05 required several updates before reaching minima.
- input layer : 20 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
- output layer : 15 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
- hidden layer : a 20x15 matrix with weights drawn from a Gaussian normal distribution.
- bias : the input layer, hidden layer and output layers have bias drawn randomly from a normal distribution and scaled down by 0.001
- number of hidden layers : 2
- activation function : tanh
- output layer function : sigmoid
- number of epochs = 200 : optimun value since a value lesser than 200 wasn't enough for the model to learn, whereas a value greater than 200 overfit.
- error function : mean squared error (MSE)
- train-test split : 80%-20%


### Requirements
- pandas
- numpy
- matplotlib

### Run
Execute ```python Neural_Net.py```

Implemented in Python.
