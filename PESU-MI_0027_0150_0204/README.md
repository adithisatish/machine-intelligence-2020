# Designing a Neural Network
## Machine Intelligence (UE18CS303) Assignment 3

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. The network implemented here is an Artificial Neural Network (ANN). 

#### Dataset

Low Birth weight (LBW) acts as an indicator of sickness in newborn babies. LBW is closely
associated with infant mortality as well as various health outcomes later in life. Various studies
show strong correlation between maternal health during pregnancy and the child’s birth weight.
Health indicators of pregnant women such as age, height, weight, community etc are very helpful 
for early detection of potential LBW cases. This detection is treated as a classification problem
between LBW and not-LBW classes, i.e. it can be treated as a binary classification problem. 

Size of raw dataset: 96 rows, 9 attributes

The attributes of the raw dataset include the community the mother belonged to, age, weight, delivery phase, haemoglobin content, 
folic acid intake (yes or no), blood pressure, educational qualification and her residence.

#### Preprocessing

The attribute "Education" had no variation in the dataset whatsoever and was hence dropped. The attribute "Delivery phase" also didn't show variation and was hence dropped as well.

The attributes "Community" and "IFA" had no missing attributes. In order to clean the dataset, boxplots were plotted for all numeric attributes (age, weight, haemoglobin content, folic acid intake and BP). This showed that all numeric attributes except for Weight (i.e. HB, BP and Age)  had outliers. For these columns, the NaNs were replaced with the attribute median. For Weight, the NaNs were replaced with the mean value. For the categorical attributes, mode replacement was done in order to impute values for the NaNs. 

The categorical variables, Community and Residence had to be encoded in order to obtain their numerical representations; done using one-hot encoding.

Size of preprocessed dataset: 96 rows, 12 attributes (11 features and 1 target)

#### Designing the Network

The ANN consists of an input layer, two hidden layers and one output layer. 

Input layer:

    Consists of one neuron per feature 
    No activation function 

First hidden layer: 

    Number of neurons = 20 
    Activation function = tanh
    Input vector dimensions: (1, 11)
    Weight matrix dimensions: (11,20)
    Bias vector dimensions: (1, 20)

Second hidden layer:

    Number of neurons = 15
    Activation function = tanh
    Input vector dimensions: (1, 20)
    Weight matrix dimensions: (20, 15)
    Bias vector dimensions: (1,15)

Output layer:

    Number of neurons = 1
    Activation function = sigmoid
    Input vector dimensions: (1, 15)
    Output dimensions: (1, 1)
    
#### Hyperparameters
Lolol

### Execution

Navigate to the \src directory.

To run the neural network, run ```python Neural_Net.py```

To preprocess the dataset, run ```python Preprocess.py```


### Authors:
- Adithi Satish
- Ananya Veeraraghavan
- Shriya Shankar
