import pandas as pd
import numpy as np
import torch
import timeit
import random
import matplotlib.pyplot as plt

import nn
import trainer
import selector

torch.set_printoptions(precision=10)

##Load dataset
data_total = pd.read_csv('3425_data_sample.csv')


print('Start to perform dataset')
best_NNs_on_single_signal = []
data_array = data_total.values.astype(np.float64)

##Basic information of current dataset
target_index = 0

## Handle outliers
mean_array = np.average(data_array,0)
from scipy import stats
mode_array = stats.mode(data_array).mode.reshape(data_array.shape[1])
for i in range(data_array.shape[1]):
    for j in range(data_array.shape[0]):
        if i==4:
            if data_array[j][i]==1:
                data_array[j][i]=4
            elif data_array[j][i]==2:
                data_array[j][i]=3
            elif data_array[j][i] == 3:
                data_array[j][i] = 2
            elif data_array[j][i] == 4:
                data_array[j][i] = 1
        if data_array[j][i]<0:
            if i==target_index:
                data_array[j][i] = mode_array[i]
            else:
                data_array[j][i] = mean_array[i]
##Modify V1 value to match prior experience




#x_array = selector.feature_selector().PCA_selector(X=np.delete(data_array[:, 2:], [16], axis=1).astype(np.float64), k=20)   ### PCA to make sense

### Split training set and testing set

train_array = data_array[:int(0.66*data_array.shape[0]),:]
test_array = data_array[int(0.66*data_array.shape[0]):,:]

x_train_array = np.concatenate((train_array[:,:target_index],train_array[:,target_index+1:]),axis=1)
y_train_array = train_array[:,target_index]
y_train_array = y_train_array - min(np.unique(y_train_array))

x_test_array = np.concatenate((test_array[:,:target_index],test_array[:,target_index+1:]),axis=1)
y_test_array = test_array[:,target_index]
y_test_array = y_test_array - min(np.unique(y_test_array))

##Determine hyperparameters of Neuron Network
input_neurons = x_train_array.shape[1]
hidden_neurons = 50
output_neurons = len(np.unique(y_train_array))
learning_rate = 0.5
num_epochs = 2000




### Transform numpy array to tensor



X_test = torch.tensor(x_test_array, dtype=torch.float)
X_train = torch.tensor(x_train_array, dtype=torch.float)

Y_test = torch.tensor(y_test_array, dtype=torch.long)
Y_train = torch.tensor(y_train_array, dtype=torch.long)

##Construct neuron network and train it
net = nn.TwoLayerNN(input_neurons, hidden_neurons, output_neurons)

loss_func = torch.nn.CrossEntropyLoss()

my_trainer = trainer.NNTrainer(num_epochs=num_epochs, learning_rate=learning_rate, loss_function=loss_func)

best_NN = my_trainer.process(NN=net, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)


"""
# Split testing set and training set randomly, but still based on fixed ratio of various depression level
depression_levels = np.unique(data_array[:, 1])
data_array = np.array(sorted(data_array, key = lambda x:x[1]))
num_samples_per_level = 4         ### Hyperparameter that controls number of samples each level in test set

x_test_array = np.empty((4*num_samples_per_level, x_array.shape[1]))
y_test_array = np.empty(4*num_samples_per_level)

raws_to_delete = []
for l in range(len(depression_levels)):
    indices = random.sample(range(0, 16), num_samples_per_level)
    for i in range(len(indices)):
        x_test_array[num_samples_per_level*l+i, :] = data_array[48*l+indices[i], 2:]
        y_test_array[num_samples_per_level*l+i] = data_array[48*l+indices[i], 1]
        raws_to_delete.append(48*l+indices[i])
data_array = np.delete(data_array, raws_to_delete, axis=0)

x_train_array = data_array[:, 2:]
y_train_array = data_array[:, 1]

X_test = torch.tensor(x_test_array.astype(np.float), dtype=torch.float)
X_train = torch.tensor(x_train_array.astype(np.float), dtype=torch.float)

Y_test = torch.tensor(y_test_array.astype(np.float), dtype=torch.long)
Y_train = torch.tensor(y_train_array.astype(np.float), dtype=torch.long)


### Start to construct and train NN
net = nn.TwoLayerNN(input_neurons, hidden_neurons, output_neurons)

loss_func = torch.nn.CrossEntropyLoss()

my_trainer = trainer.NNTrainer(num_epochs=num_epochs, learning_rate=learning_rate, loss_function=loss_func)

best_NN = my_trainer.process(NN=net, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

# NNs.append()
"""


