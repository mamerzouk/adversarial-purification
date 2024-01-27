import torch
import torch.nn as nn
from torch import optim


import argparse
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw
from helper import load_model
from ids import IDS
from diffusion import MLP, Diffusion
from attack import fgsm

def main(diffusion_epochs, diffusion_lr, diffusion_hidden_dim, noise_steps, 
         epsilon, epsilon_steps, ids_lr, ids_epochs, device, ids_hidden_dim=None):
    if device=='none':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids_hidden_dim = [256, 512, 1024, 512, 256]

    print(ids_lr)
    print(ids_epochs)
    print(device)

    ids_model = IDS(hidden_dim=ids_hidden_dim).to(device)
    ids_loss = nn.CrossEntropyLoss()
    ids_optimizer = torch.optim.Adam(ids_model.parameters(), lr=ids_lr)

    ids_log_name = "IDS_"+str(ids_loss)[:-2]+"_LR_"+str(ids_lr)+"_E_"+str(ids_epochs)+"_H_"+str(ids_hidden_dim).replace(", ", "-")+"_"+device

    x_train, y_train, x_test, y_test = preprocess_unsw()
    # Convert the data to PyTorch Tensor in the GPU
    x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)

    _ = load_model(ids_log_name, ids_model)

    x_test_adv = fgsm(model=ids_model, loss=ids_loss, optimizer=ids_optimizer, epsilon=epsilon, epsilon_steps=epsilon_steps, x_test=x_test, y_test=y_test, log_name=ids_log_name)

    diffusion_model = MLP(data_dim=196, hidden_dim=diffusion_hidden_dim, emb_dim=256, device=device).to(device)
    optimizer = optim.AdamW(diffusion_model.parameters(), lr=diffusion_lr)
    diffusion_loss = nn.MSELoss()
    diffusion_process = Diffusion(data_size=196, noise_steps=noise_steps, device=device)

    diffusion_log_name = "DIFFUSION_"+str(diffusion_loss)[:-2]+"_LR_"+str(diffusion_lr)+"_T_"+str(noise_steps)+"_E_"+str(diffusion_epochs)+"_H_"+str(10)+"-"+str(diffusion_hidden_dim)+"_"+device

    _ = load_model(diffusion_log_name, diffusion_model)


if __name__ == "__main__":

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", required=True, default="none", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("-ie", "--ids_epochs", required=True, default=100, type=int, help="Number of epochs to train the model.")
    parser.add_argument("-il", "--ids_learning_rate", required=True, default=1e-3, type=float, help="Learning rate")
    parser.add_argument("-s", "--epsilon", required=True, default=0.001, type=float, help="Epsilon the adversarial perturbation amplitude.")
    parser.add_argument("-t", "--epsilon_steps", required=True, default=31, type=int, help="Number of steps in epsilon.")
    parser.add_argument("-de", "--diffusion_epochs", required=True, default=100, type=int, help="Number of epochs to train the model.")
    parser.add_argument("-dl", "--diffusion_learning_rate", required=True, default=1e-3, type=float, help="Learning rate")
    parser.add_argument("-n", "--noise_steps", required=True, default=1000, type=int, help="Noise steps")
    parser.add_argument("-di", "--diffusion_hidden_dim", required=True, default=1024, type=int, help="Dimension of hidden layer.")
    args = parser.parse_args()

    main(diffusion_epochs=args.diffusion_epochs, diffusion_lr=args.diffusion_lr, diffusion_hidden_dim=args.diffusion_hidden_dim, noise_steps=args.noise_steps, epsilon=args.epsilon, epsilon_steps=args.epsilon_steps, ids_lr=args.ids_learning_rate, ids_epochs=args.ids_epochs, device=args.device)
