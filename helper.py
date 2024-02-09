import pickle
from matplotlib import pyplot as plt
import torch

def save_model(log_name, model, logs):
    # Save the model parameters into a file
    torch.save(model.state_dict(), "./results/"+log_name+".pytorch")

    # Save the accuracy and loss logs into a file
    with open("./results/"+log_name+".logs", 'wb') as file:
        pickle.dump(logs, file)

def load_model(log_name, model):
    # Load the model parameters from a file
    model.load_state_dict(torch.load("./results/"+log_name+".pytorch"))

    # Load the accuracy and loss logs from a file
    with open("./results/"+log_name+".logs", 'rb') as file:
        logs = pickle.load(file)
    return logs

def accuracy(pred, y):
    return (pred.argmax(dim=1) == y).type(torch.float).sum().item()/pred.shape[0]
