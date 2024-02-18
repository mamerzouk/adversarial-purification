import sys
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw, preprocess_kdd
from helper import load_model, accuracy
from ids import IDS
from diffusion import MLP, Diffusion


def main(dataset, diffusion_epochs, diffusion_lr, diffusion_hidden_dim, noise_steps,
         epsilon, epsilon_steps, ids_lr, ids_epochs, beta_start, beta_end,
         device, ids_hidden_dim=None, reconstruction_curve=False, reconstruction_step=1,
         generate_adversarial=False):
    if device=='none':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if dataset == 'UNSW-NB15':
        x_train, y_train, x_test, y_test = preprocess_unsw()
    if dataset == 'NSL-KDD':
        x_train, y_train, x_test, y_test = preprocess_kdd()    
        
    # Convert the data to PyTorch Tensor in the GPU
    x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)

    ids_model = IDS(input_dim=x_train.shape[1], hidden_dim=ids_hidden_dim).to(device)
    ids_loss = torch.nn.CrossEntropyLoss()
    ids_optimizer = torch.optim.Adam(ids_model.parameters(), lr=ids_lr)

    ids_log_name = "IDS_" + dataset + "_" + str(ids_loss)[:-2] + \
                    "_LR_" + str(ids_lr) + \
                    "_E_" + str(ids_epochs) + \
                    "_H_" + str(ids_hidden_dim).replace(", ", "-") + \
                    "_" + device

    print(f"Loading {ids_log_name} ...")
    _ = load_model(ids_log_name, ids_model)

    print("Loading adversarial examples ...")
    if generate_adversarial:
        from attack import fgsm
        x_test_adv = fgsm(model=ids_model,
                        loss=ids_loss,
                        optimizer=ids_optimizer,
                        epsilon=epsilon,
                        epsilon_steps=epsilon_steps,
                        x_test=x_test,
                        y_test=y_test,
                        log_name=ids_log_name)
    with open('results/ADV_'+ids_log_name+'.np', 'rb') as file:
        x_test_adv = np.load(file)

    x_test_adv = torch.Tensor(x_test_adv).to(device)

    diffusion_model = MLP(data_dim=x_train.shape[1],
                          hidden_dim=diffusion_hidden_dim,
                          emb_dim=256,
                          device=device).to(device)
    #diffusion_optimizer = optim.AdamW(diffusion_model.parameters(), lr=diffusion_lr) #keep for art
    diffusion_loss = torch.nn.MSELoss()
    diffusion_process = Diffusion(data_size=196, noise_steps=noise_steps, device=device)

    diffusion_log_name = "DIFFUSION_" + dataset + "_T_" + str(noise_steps) + \
                            "_B_" + str(beta_start)+"_" + str(beta_end) + \
                            "_" + str(diffusion_loss)[:-2] + \
                            "_LR_" + str(diffusion_lr) + \
                            "_E_" + str(diffusion_epochs) + \
                            "_H_" + str(10) + "-" + str(diffusion_hidden_dim)+\
                            "_"+device

    print(f"Loading {diffusion_log_name} ...")
    _ = load_model(diffusion_log_name, diffusion_model)

    if reconstruction_curve:
        print("Plotting the reconstruction curve ...")

        train_loss = [0]
        test_loss = [0]
        adv_loss = [0]
        test_adv_loss = [0]
        train_acc = [accuracy(ids_model.forward(x_train), y_train)]
        test_acc = [accuracy(ids_model.forward(x_test), y_test)]
        adv_acc = [accuracy(ids_model.forward(x_test_adv), y_test)]

        pbar = tqdm(range(1, noise_steps+1, reconstruction_step))
        for t in pbar:
            if t > 100:
                if t % 100 != 0:
                    continue
            
            diffusion_model.eval()
            with torch.no_grad():
                ts = torch.ones(x_train.shape[0]).int().to(device) * (t-1)
                x_t, _ = diffusion_process.noise_data(x_train, ts)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model,
                                                                x_t,
                                                                t,
                                                                progress_bar=False)
                loss = diffusion_loss(x_train, reconstructed_x)
                train_loss.append(loss.item())
                pred = ids_model.forward(reconstructed_x)
                train_acc.append(accuracy(pred, y_train))

                ts = torch.ones(x_test.shape[0]).int().to(device) * (t-1)
                x_t, _ = diffusion_process.noise_data(x_test, ts)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model,
                                                                x_t,
                                                                t,
                                                                progress_bar=False)
                loss = diffusion_loss(x_test, reconstructed_x)
                test_loss.append(loss.item())
                pred = ids_model.forward(reconstructed_x)
                test_acc.append(accuracy(pred, y_test))

                ts = torch.ones(x_test_adv.shape[0]).int().to(device) * (t-1)
                x_t, _ = diffusion_process.noise_data(x_test_adv, ts)
                reconstructed_x = diffusion_process.reconstruct(diffusion_model,
                                                                x_t,
                                                                t,
                                                                progress_bar=False)
                loss = diffusion_loss(x_test_adv, reconstructed_x)
                adv_loss.append(loss.item())
                loss = diffusion_loss(x_test, reconstructed_x)
                test_adv_loss.append(loss.item())
                pred = ids_model.forward(reconstructed_x)
                adv_acc.append(accuracy(pred, y_test))

            if t%10 == 0:
                with open("./results/progress_reconstruction_"+diffusion_log_name+".logs",
                            'wb') as file:
                    pickle.dump((train_loss,
                                    test_loss,
                                    adv_loss,
                                    test_adv_loss,
                                    train_acc,
                                    test_acc,
                                    adv_acc), file)

        with open("./results/reconstruction_"+diffusion_log_name+".logs", 'wb') as file:
            pickle.dump((train_loss, test_loss, adv_loss, test_adv_loss, train_acc, test_acc, adv_acc), file)
        max_adv_acc_step = adv_acc.index(max(adv_acc))
        max_adv_acc_beta = beta_start + ((beta_end-beta_start)/noise_steps) * max_adv_acc_step
        print(f"Maximum accuracy on adversarial data is{adv_acc[max_adv_acc_step]}\
              at noise step{max_adv_acc_step}\
              where variance is {max_adv_acc_beta}")
    else:
        t = noise_steps
        diffusion_model.eval()
        with torch.no_grad():
            ts = torch.ones(x_train.shape[0]).int().to(device) * (t-1)
            x_t, _ = diffusion_process.noise_data(x_train, ts)
            reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
            loss = diffusion_loss(x_train, reconstructed_x)
            print(f"Reconstruction loss on the training set : {loss.item()}")
            pred = ids_model.forward(reconstructed_x)
            print(f"Accuracy on the reconstructed training set : {accuracy(pred, y_train)}")

            ts = torch.ones(x_test.shape[0]).int().to(device) * (t-1)
            x_t, _ = diffusion_process.noise_data(x_test, ts)
            reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
            loss = diffusion_loss(x_test, reconstructed_x)
            print(f"Reconstruction loss on the testing set : {loss.item()}")
            pred = ids_model.forward(reconstructed_x)
            print(f"Accuracy on the reconstructed testing set : {accuracy(pred, y_test)}")

            ts = torch.ones(x_test_adv.shape[0]).int().to(device) * (t-1)
            x_t, _ = diffusion_process.noise_data(x_test_adv, ts)
            reconstructed_x = diffusion_process.reconstruct(diffusion_model, x_t, t)
            loss = diffusion_loss(x_test, reconstructed_x)
            print(f"Reconstruction loss on the adversarial testing set : {loss.item()}")
            pred = ids_model.forward(reconstructed_x)
            print(f"Accuracy on reconstructed adversarial testing set : {accuracy(pred, y_test)}")


if __name__ == "__main__":

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--device",
                        required=True,
                        default="none",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("-ie",
                        "--ids_epochs",
                        required=True,
                        default=100,
                        type=int,
                        help="Number of epochs to train the model.")
    parser.add_argument("-il",
                        "--ids_learning_rate",
                        required=True,
                        default=1e-3,
                        type=float,
                        help="Learning rate")
    parser.add_argument("-s",
                        "--epsilon",
                        required=True,
                        default=0.001,
                        type=float,
                        help="Epsilon the adversarial perturbation amplitude.")
    parser.add_argument("-t",
                        "--epsilon_steps",
                        required=True,
                        default=31,
                        type=int,
                        help="Number of steps in epsilon.")
    parser.add_argument("-de",
                        "--diffusion_epochs",
                        required=True,
                        default=100,
                        type=int,
                        help="Number of epochs to train the model.")
    parser.add_argument("-dl",
                        "--diffusion_learning_rate",
                        required=True,
                        default=1e-3,
                        type=float,
                        help="Learning rate")
    parser.add_argument("-n",
                        "--noise_steps",
                        required=True,
                        default=1000,
                        type=int,
                        help="Noise steps")
    parser.add_argument("-di",
                        "--diffusion_hidden_dim",
                        required=True,
                        default=1024,
                        type=int,
                        help="Dimension of hidden layer.")
    parser.add_argument("-rc",
                        "--reconstruction_curve",
                        required=False,
                        default=0,
                        type=int,
                        help="Reconstruction curve")
    parser.add_argument("-rs",
                        "--reconstruction_step",
                        required=False,
                        default=1,
                        type=int,
                        help="Reconstruction step")
    parser.add_argument("-bs",
                        "--beta_start",
                        required=True,
                        default=1e-4,
                        type=float,
                        help="Start value for beta")
    parser.add_argument("-be",
                        "--beta_end",
                        required=True,
                        default=0.02,
                        type=float,
                        help="Start value for beta")
    parser.add_argument("-ga",
                        "--generate_adversarial",
                        required=False,
                        default=0,
                        type=int,
                        help="Generate adversarial examples")
    parser.add_argument("-ds",
                        "--dataset",
                        required=False,
                        default='UNSW-NB15',
                        type=str,
                        help="Dataset")
    parser.add_argument("-ihd",
                        "--ids_hidden_dim",
                        required=True,
                        default=[],
                        nargs='+',
                        type=int,
                        help="Train and test")
    args = parser.parse_args()

    main(diffusion_epochs=args.diffusion_epochs,
         diffusion_lr=args.diffusion_learning_rate,
         diffusion_hidden_dim=args.diffusion_hidden_dim,
         noise_steps=args.noise_steps,
         epsilon=args.epsilon,
         epsilon_steps=args.epsilon_steps,
         ids_lr=args.ids_learning_rate,
         ids_epochs=args.ids_epochs,
         device=args.device,
         reconstruction_curve=args.reconstruction_curve,
         reconstruction_step=args.reconstruction_step,
         beta_start=args.beta_start,
         beta_end=args.beta_end,
         generate_adversarial=args.generate_adversarial,
         dataset=args.dataset,
         ids_hidden_dim=args.ids_hidden_dim)
