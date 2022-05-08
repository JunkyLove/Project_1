import torch
import random


class x_y_shuffler:
    def shuffle(self, x, y):
        index = [i for i in range(x.size()[0])]
        random.shuffle(index)
        new_x = torch.empty(x.size())
        new_y = torch.empty(y.size())
        for i in range(len(index)):
            new_x[i, :] = x[index[i], :]
            new_y[i] = y[index[i]]
        new_y = new_y.long()
        return new_x, new_y






