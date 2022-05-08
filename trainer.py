import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

import nn
import shuffler


class NNTrainer:
    def __init__(self, num_epochs, learning_rate, loss_function):
        self.epochs = num_epochs
        self.lr = learning_rate
        self.loss_func = loss_function

    def process(self, NN, x_train, y_train, x_test, y_test):
        best_NN = copy.deepcopy(NN)
        best_test_precision = 0
        all_losses = []

        ### Selection of different optimiser
        #optimiser = torch.optim.SGD(NN.parameters(), lr=self.lr)
        #optimiser = torch.optim.SGD(NN.parameters(), lr=self.lr, momentum=0.9)
        optimiser = torch.optim.Adam(NN.parameters())

        x_train, y_train = shuffler.x_y_shuffler().shuffle(x_train, y_train)
        x_test, y_test = shuffler.x_y_shuffler().shuffle(x_test, y_test)

        for epoch in range(self.epochs + 1):
            #NN.train()
            y_pred = NN(x_train)
            loss = self.loss_func(y_pred, y_train)
            all_losses.append(loss.item())

            #NN.eval()
            test_y_pred = NN(x_test)
            test_predicted = torch.max(F.softmax(test_y_pred, 1), 1)[1]
            test_total = test_predicted.size(0)
            test_correct = test_predicted.data.numpy() == y_test.data.numpy()
            test_precision = sum(test_correct) / test_total

            if test_precision > best_test_precision:
                best_test_precision = test_precision
                best_NN = copy.deepcopy(NN)
                print('A better NN was found. Current testing accuracy is %.2f %%'
                      % (100 * sum(test_correct) / test_total))

            if epoch % 500 == 0:
                predicted = torch.max(F.softmax(y_pred, 1), 1)[1]

                total = predicted.size(0)
                correct = predicted.data.numpy() == y_train.data.numpy()

                print('Epoch [%d/%d] Loss: %.4f   Training Accuracy: %.2f %%'
                      % (epoch + 1, self.epochs, loss.item(), 100 * sum(correct) / total))
            if epoch % 100 == 0:
                print('*******  Epoch [%d/%d], Testing Accuracy: %.2f %%  **********'
                      % (epoch + 1, self.epochs, 100 * sum(test_correct) / test_total))

            #NN.train()
            NN.zero_grad()
            loss.backward()
            optimiser.step()

        plt.figure()
        plt.plot(all_losses)
        plt.show()

        ### Print Best-NN confusion matrix
        confusion = torch.zeros(NN.num_out_neurons, NN.num_out_neurons)
        test_y_pred = best_NN(x_test)
        test_predicted = torch.max(F.softmax(test_y_pred, 1), 1)[1]

        for i in range(x_test.shape[0]):
            actual_class = y_test.data[i]
            predicted_class = test_predicted.data[i]

            confusion[actual_class][predicted_class] += 1

        print('')
        print('Best-NN Confusion matrix for testing:')
        print(confusion.numpy())

        return best_NN
