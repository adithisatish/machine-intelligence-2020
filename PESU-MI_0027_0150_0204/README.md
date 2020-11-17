# Designing a Neural Network
## Machine Intelligence (UE18CS303) Assignment 3

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. The network implemented here is an Artificial Neural Network (ANN). 

### Execution

Navigate to the \src directory.

To run the neural network, run ```python Neural_Net.py```

To preprocess the dataset, run ```python Preprocess.py```

### Problem Statement and Approach

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

Class NN:
- The __init__ method takes two parameters:
  - num_features: The number of features the dataset has (this is required to set dimensions for the first hidden layer)
  - dims: A list of size 2, representing the number of neurons in each of the two hidden layers respectively
- __NN.fit__ is the method used to compile and train the model using forward propagation to calculate the predicted value, find the error using __MSE__ (Mean Squared Error) and then perform back propagation using stochastic gradient descent in order to get optimum weights. Once trained, the weights are at their optimum level and new data can be passed to the predict function. The fit method takes 3 parameters:
  - X: The matrix of features
  - Y: The target vector
  - epochs: The number of epochs to use
- __NN.predict__ is the method used after training (fit) in order to predict the values for a new instance(s). It does a simple feed forward of the input vector linearly combined with the optimum weights and biases, resulting in a probability (as the output layer is a sigmoid function). The method takes one parameter:
  - X: The test instance(s)

Once the predicted probability is returned, it is compared against a set threshold of 0.6. Any result greater than the threshold is declared as 1 and any result lower than the threshold is assigned the value 0. 

Hyperparameters:
- learning rate = 0.05 : a value larger than 0.05 led to divergence whereas a value lesser than 0.05 required several updates before reaching minima. !!!!!ADD PART ABOUT DECAY HERE!!!!
- input layer : consists of a neuron for each feature, i.e 11 neurons
- 1st hidden layer: 20 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
- 2nd hidden layer : a 20x15 matrix with weights drawn from a Gaussian normal distribution.
- output layer : 15 neurons with weights from randomly assigned from a normal distribution with mean=0, standard deviation=1
- bias : the input layer, hidden layers and output layers have bias drawn randomly from a normal distribution and scaled down by 0.001
- number of hidden layers : 2
- activation function : tanh
- output layer function : sigmoid
- number of epochs = 200 : optimun value since a value lesser than 200 wasn't enough for the model to learn, whereas a value greater than 200 overfit.
- error function : mean squared error (MSE)
- train-test split : 80%-20%

The ANN consists of an input layer, two hidden layers and one output layer. Detailed explanation -

Input layer:

    Number of neurons = number of features = 11
    Activation function = None

First hidden layer: 

    Fully connected
    Number of neurons = 20 
    Activation function = tanh
    Input vector dimensions: (1, 11)
    Weight matrix dimensions: (11,20)
    Bias vector dimensions: (1, 20)

Second hidden layer:

    Fully connected
    Number of neurons = 15
    Activation function = tanh
    Input vector dimensions: (1, 20)
    Weight matrix dimensions: (20, 15)
    Bias vector dimensions: (1, 15)

Output layer:

    Number of neurons = 1
    Activation function = sigmoid
    Input vector dimensions: (1, 15)
    Weight matrix dimensions: (15, 1)
    Bias vector dimensions: (1, 1)
    Output dimensions: (1, 1)

#### Performance Metrics

A train-test split of 80-20% was chosen to fit this model. This is a very common split and has been proven to give optimum performance metrics for various models, and held up with the ANN as well. 

Number of epochs: 200

Training Performance Metrics:
    
    Accuracy: 89.47%
    Precision: 92.59%
    Recall: 92.59%
    F1 Score: 0.9259

Testing Performance Metrics:

    Accuracy: 85%
    Precision: 94.11%
    Recall: 88.88%
    F1 Score: 0.9142

The time taken to train the model, on average was found to be = 1.58s


### Authors:
- Adithi Satish
- Ananya Veeraraghavan
- Shriya Shankar
