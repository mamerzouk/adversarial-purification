import sys
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw, preprocess_kdd
from helper import save_model, accuracy

# Build a neural network for intrusion detection with a variable number of hidden layers and units
class IDS(torch.nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None):
        super().__init__()
        self.stack = torch.nn.Sequential()
        self.stack.add_module("hidden_linear_0",
                              torch.nn.Linear(input_dim, hidden_dim[0]))
        self.stack.add_module("hidden_activation_0", torch.nn.ReLU())
        if hidden_dim is None:
            hidden_dim= []
        for i in range(len(hidden_dim)-1):
            self.stack.add_module("hidden_linear_"+str(i+1),
                                  torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            self.stack.add_module("hidden_activation_"+str(i+1),torch.nn.ReLU())
        self.stack.add_module("output_linear", torch.nn.Linear(hidden_dim[-1], 2))

    def forward(self, x):
        logits = self.stack(x)
        return logits #F.softmax(logits, dim=1).squeeze()

# Train the IDS and progressively show the learning curve
def train(model,
          loss_fn,
          optimizer,
          x_train,
          y_train,
          x_test,
          y_test,
          epochs=1000,
          train_and_test=False):

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    pbar = tqdm(range(epochs))
    for _ in pbar:
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        train_loss.append(loss.item())
        train_acc.append(accuracy(pred, y_train))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if train_and_test:
            with torch.no_grad():
                pred = model(x_test)
                loss = loss_fn(pred, y_test)
                test_loss.append(loss.item())
                test_acc.append(accuracy(pred, y_test))
            pbar.set_postfix(CEL_TRAIN=train_loss[-1], CEL_TEST=test_loss[-1],
                             ACC_TRAIN=train_acc[-1], ACC_TEST=test_acc[-1])
        else:
            pbar.set_postfix(CEL_TRAIN=train_loss[-1], ACC_TRAIN=train_acc[-1])

    return train_loss, train_acc, test_loss, test_acc

def main(dataset, lr, epochs, train_and_test=False, device='none', hidden_dim=None):
    if device=='none':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dataset == 'UNSW-NB15':
        x_train, y_train, x_test, y_test = preprocess_unsw()
    elif dataset == 'NSL-KDD':
            x_train, y_train, x_test, y_test = preprocess_kdd()

    # Convert the data to PyTorch Tensor in the GPU
    x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)

    model = IDS(input_dim=x_train.shape[1], hidden_dim=hidden_dim).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_name = "IDS_" + dataset + "_" + str(loss)[:-2] + \
            "_LR_"+str(lr) + \
            "_E_" + str(epochs) + \
            "_H_" + str(hidden_dim).replace(", ", "-") + \
            "_"+device

    print(f"Training {log_name} ...")
    train_loss, train_acc, test_loss, test_acc = train(model=model,
                                                       loss_fn=loss,
                                                       optimizer=optimizer,
                                                       x_train=x_train,
                                                       y_train=y_train,
                                                       x_test=x_test,
                                                       y_test=y_test,
                                                       epochs=epochs,
                                                       train_and_test=train_and_test)

    logs = (train_acc, train_loss, test_acc, test_loss)
    save_model(log_name, model, logs)

if __name__ == "__main__":

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--device",
                        required=True,
                        default="none",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("-e",
                        "--epochs",
                        required=True,
                        default=100,
                        type=int,
                        help="Number of epochs to train the model.")
    parser.add_argument("-l",
                        "--learning_rate",
                        required=True,
                        default=1e-3,
                        type=float,
                        help="Learning rate")
    parser.add_argument("-ds",
                        "--dataset",
                        required=True,
                        default="UNSW-NB15",
                        type=str,
                        help="Dataset")
    parser.add_argument("-tt",
                        "--train_and_test",
                        required=False,
                        default=0,
                        type=int,
                        help="Train and test")
    parser.add_argument("-hd",
                        "--hidden_dim",
                        required=True,
                        default=[],
                        nargs='+',
                        type=int,
                        help="Train and test")

    args = parser.parse_args()

    main(lr=args.learning_rate, epochs=args.epochs, device=args.device, dataset=args.dataset, train_and_test=args.train_and_test, hidden_dim=args.hidden_dim)
