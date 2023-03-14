# In this folder is the construction of a predictive model using the SICP feature set alone

## The code of our deep model part is divided into two steps as a whole :

The preparation work is to divide the data set into training set and test set, and then divide the training set into training set and verification set.

### 1、First step -- construct_model:
Firstly, the model is constructed and some attributes and parameters of the deep model are determined. The training set and the validation set are used to determine the hyperparameters, such as how many layers, the convolution kernel of each layer and the learning rate.

### 2、Second step -- test_model:
After these hyperparameters are determined, the model is tested, the model is run with the training set, the model is stopped early with the verification set, and the model is stopped early with the test set.
