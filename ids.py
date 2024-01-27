import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

import argparse
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw
from helper import save_model, plot_curve, accuracy

# Build a neural network for intrusion detection with a variable number of hidden layers and units
class IDS(nn.Module):
    def __init__(self, hidden_dim=[256]):
        super().__init__()
        self.stack = nn.Sequential()
        self.stack.add_module("hidden_linear_0", nn.Linear(196, hidden_dim[0]))
        self.stack.add_module("hidden_activation_0", nn.ReLU())
        for i in range(len(hidden_dim)-1):
          self.stack.add_module("hidden_linear_"+str(i+1), nn.Linear(hidden_dim[i], hidden_dim[i+1]))
          self.stack.add_module("hidden_activation_"+str(i+1), nn.ReLU())
        self.stack.add_module("output_linear", nn.Linear(hidden_dim[-1], 2))

    def forward(self, x):
        logits = self.stack(x)
        return logits #F.softmax(logits, dim=1).squeeze()
    
# Train the IDS and progressively show the learning curve
def train(model, loss_fn, optimizer, x_train, y_train, x_test, y_test, epochs=1000, train_and_test=False):

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []


    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        train_loss.append(loss.item())
        train_acc.append(accuracy(pred, y_train))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if train_and_test:
            print(train_and_test)
            with torch.no_grad():
                pred = model(x_test)
                loss = loss_fn(pred, y_test)
                test_loss.append(loss.item())
                test_acc.append(accuracy(pred, y_test))
            pbar.set_postfix(CEL_TRAIN=train_loss[-1], CEL_TEST=test_loss[-1], ACC_TRAIN=train_acc[-1], ACC_TEST=test_acc[-1])
        else:
            pbar.set_postfix(CEL_TRAIN=train_loss[-1], ACC_TRAIN=train_acc[-1])
            

    return train_loss, train_acc, test_loss, test_acc
    


def main(lr, epochs, device, hidden_dim=None):
    if device=='none':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = [256, 512, 1024, 512, 256]

    print(lr)
    print(epochs)
    print(device)

    model = IDS(hidden_dim=hidden_dim).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_name = "IDS_"+str(loss)[:-2]+"_LR_"+str(lr)+"_E_"+str(epochs)+"_H_"+str(hidden_dim).replace(", ", "-")+"_"+device

    x_train, y_train, x_test, y_test = preprocess_unsw()
    # Convert the data to PyTorch Tensor in the GPU
    x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)

    train_loss, train_acc, test_loss, test_acc = train(model=model, loss_fn=loss, optimizer=optimizer, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs)

    logs = (train_acc, train_loss, test_acc, test_loss)
    save_model(log_name, model, logs)
    plot_curve(log_name, blue=train_acc, dotted_blue=train_loss, orange=test_acc, dotted_orange=test_loss)

if __name__ == "__main__":

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", required=True, default="none", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("-e", "--epochs", required=True, default=100, type=int, help="Number of epochs to train the model.")
    parser.add_argument("-l", "--learning_rate", required=True, default=1e-3, type=float, help="Learning rate")
    
    args = parser.parse_args()

    main(lr=args.learning_rate, epochs=args.epochs, device=args.device)



    