import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
import pickle

import argparse
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw
from helper import load_model, plot_curve, accuracy
from ids import IDS
from diffusion import MLP, Diffusion
from attack import fgsm

def main(diffusion_epochs, diffusion_lr, diffusion_hidden_dim, noise_steps, 
         epsilon, epsilon_steps, ids_lr, ids_epochs, beta_start, beta_end,
         device, ids_hidden_dim=None, reconstruction_curve=False):
    if device=='none':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids_hidden_dim = [256, 512, 1024, 512, 256]

    ids_model = IDS(hidden_dim=ids_hidden_dim).to(device)
    ids_loss = nn.CrossEntropyLoss()
    ids_optimizer = torch.optim.Adam(ids_model.parameters(), lr=ids_lr)

    ids_log_name = "IDS_"+str(ids_loss)[:-2]+"_LR_"+str(ids_lr)+"_E_"+str(ids_epochs)+"_H_"+str(ids_hidden_dim).replace(", ", "-")+"_"+device

    x_train, y_train, x_test, y_test = preprocess_unsw()
    # Convert the data to PyTorch Tensor in the GPU
    x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)

    print("Loading {} ...".format(ids_log_name))
    _ = load_model(ids_log_name, ids_model)

    print("Generating adversarial examples ...")
    x_test_adv = fgsm(model=ids_model, loss=ids_loss, optimizer=ids_optimizer, epsilon=epsilon, epsilon_steps=epsilon_steps, x_test=x_test, y_test=y_test, log_name=ids_log_name)
    x_test_adv = torch.Tensor(x_test_adv).to(device)

    diffusion_model = MLP(data_dim=196, hidden_dim=diffusion_hidden_dim, emb_dim=256, device=device).to(device)
    #diffusion_optimizer = optim.AdamW(diffusion_model.parameters(), lr=diffusion_lr)
    diffusion_loss = nn.MSELoss()
    diffusion_process = Diffusion(data_size=196, noise_steps=noise_steps, device=device)

    diffusion_log_name = "DIFFUSION"+"_T_"+str(noise_steps)+"_B_"+str(beta_start)+"_"+str(beta_end)+"_"+str(diffusion_loss)[:-2]+"_LR_"+str(diffusion_lr)+"_E_"+str(diffusion_epochs)+"_H_"+str(10)+"-"+str(diffusion_hidden_dim)+"_"+device

    print("Loading {} ...".format(diffusion_log_name))
    _ = load_model(diffusion_log_name, diffusion_model)

    if reconstruction_curve:
        train_loss = []
        test_loss = []
        adv_loss = []

        pbar = tqdm(range(1, noise_steps+1))
        for t in pbar:
            diffusion_model.eval()
            with torch.no_grad():
                ts = torch.ones(x_train.shape[0]).int().to(device) * (t-1)
                x_t, noise = diffusion_process.noise_data(x_train, ts)
                #predicted_noise = diffusion_model(x_t, t)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t, progress_bar=False)
                loss = diffusion_loss(x_train, reconstructed_x)
                train_loss.append(loss.item())

                ts = torch.ones(x_test.shape[0]).int().to(device) * (t-1)
                x_t, noise = diffusion_process.noise_data(x_test, ts)
                #predicted_noise = diffusion_model(x_t, t)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t, progress_bar=False)
                loss = diffusion_loss(x_test, reconstructed_x)
                test_loss.append(loss.item())
                
                ts = torch.ones(x_test_adv.shape[0]).int().to(device) * (t-1)
                x_t, noise = diffusion_process.noise_data(x_test_adv, ts)
                #predicted_noise = diffusion_model(x_t, t)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t, progress_bar=False)
                loss = diffusion_loss(x_test_adv, reconstructed_x)
                adv_loss.append(loss.item())

                if t%10 == 0:
                    plot_curve('progress_reconstruction_'+diffusion_log_name, blue=train_loss, orange=test_loss, red=adv_loss, ylim=0.05)
        plot_curve('reconstruction_'+diffusion_log_name, blue=train_loss, orange=test_loss, red=adv_loss, ylim=0.1)
        with open("./results/reconstruction_"+diffusion_log_name+".logs", 'wb') as file:
            pickle.dump((train_loss, test_loss, adv_loss), file)


    t = noise_steps
    diffusion_model.eval()
    with torch.no_grad():
        ts = torch.ones(x_train.shape[0]).int().to(device) * (t-1)
        x_t, noise = diffusion_process.noise_data(x_train, ts)
        #predicted_noise = diffusion_model(x_t, t)
        reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
        loss = diffusion_loss(x_train, reconstructed_x)
        print("Reconstruction loss on the training set : {}".format(loss.item()))
        pred = ids_model(reconstructed_x)
        print("Accuracy on the reconstructed training set : {}".format(accuracy(pred, y_train)))

        ts = torch.ones(x_test.shape[0]).int().to(device) * (t-1)
        x_t, noise = diffusion_process.noise_data(x_test, ts)
        #predicted_noise = diffusion_model(x_t, t)
        reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
        loss = diffusion_loss(x_test, reconstructed_x)
        print("Reconstruction loss on the testing set : {}".format(loss.item()))
        pred = ids_model(reconstructed_x)
        print("Accuracy on the reconstructed testing set : {}".format(accuracy(pred, y_test)))

        ts = torch.ones(x_test_adv.shape[0]).int().to(device) * (t-1)
        x_t, noise = diffusion_process.noise_data(x_test_adv, ts)
        #predicted_noise = diffusion_model(x_t, t)
        reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
        loss = diffusion_loss(x_test, reconstructed_x)
        print("Reconstruction loss on the adversarial testing set : {}".format(loss.item()))
        pred = ids_model(reconstructed_x)
        print("Accuracy on the reconstructed adversarial testing set : {}".format(accuracy(pred, y_test)))









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
    parser.add_argument("-rc", "--reconstruction_curve", required=False, default=0, type=int, help="Reconstruction curve")
    parser.add_argument("-bs", "--beta_start", required=True, default=1e-4, type=float, help="Start value for beta")
    parser.add_argument("-be", "--beta_end", required=True, default=0.02, type=float, help="Start value for beta")
    args = parser.parse_args()

    main(diffusion_epochs=args.diffusion_epochs, diffusion_lr=args.diffusion_learning_rate, diffusion_hidden_dim=args.diffusion_hidden_dim, noise_steps=args.noise_steps,
         epsilon=args.epsilon, epsilon_steps=args.epsilon_steps, ids_lr=args.ids_learning_rate, ids_epochs=args.ids_epochs, device=args.device,
         reconstruction_curve=args.reconstruction_curve, beta_start=args.beta_start, beta_end=args.beta_end)
