
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score



class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparameters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparameters = dict of hyperparameters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparameters': hyperparameters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, path):
        # Finish this function so that it loads a model and return the appropriate variables

        checkpoint = torch.load(path)
        return checkpoint


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparameters):
        # Finish this function so that it set up model then trains and saves it.
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.model_class = model_class


        epochs = hyperparameters['epoch']
        batch_size = hyperparameters['batch_size']
        num_classes = hyperparameters['num_classes']
        learning_rate = hyperparameters['learning_rate']
        sequence_length = hyperparameters['sequence_length']
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        input_size = hyperparameters['input_size']

        output_size = 4

        self.device = torch.device("cuda:2")

        train_batches = Batcher(train_X, train_y, device, batch_size=batch_size, iteration=epochs)

        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  




    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass





class Batcher:
    def __innit__(self, X, y, device, batch_size=50, iteration=None):
        self.X = X
        self.Y = Y
        self.device = device
        self.batch_size = batch_size
        self.max_iter=max_iter
        self.curr_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.size()[0], device=self.device)
        permX = self.X[permunation]
        permy = self.y[permunation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)

        self.curr_iter =+ 1

        return zip(splitX, splity)








