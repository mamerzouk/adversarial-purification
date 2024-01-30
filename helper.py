from matplotlib import pyplot as plt
import pickle
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
    

def plot_curve(log_name, blue=[], orange=[], dotted_blue=[], dotted_orange=[], red=[], dotted_red=[], x=None):
    plt.plot(red, color='tab:red')
    plt.plot(dotted_red, color='tab:red', linestyle='dashed')
    plt.plot(blue, color='tab:blue')
    plt.plot(dotted_blue, color='tab:blue', linestyle='dashed')
    plt.plot(orange, color='tab:orange')
    plt.plot(dotted_orange, color='tab:orange', linestyle='dashed')
    #plt.xlim([0, len(train_loss)])
    plt.ylim([0, 1.2])
    plt.grid()
    plt.savefig("./results/"+log_name+'.pdf',
                bbox_inches='tight', 
                transparent=True,
                pad_inches=0)
    

def accuracy(pred, y):
    return (pred.argmax(dim=1) == y).type(torch.float).sum().item()/pred.shape[0]
