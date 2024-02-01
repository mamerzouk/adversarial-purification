import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


from tqdm import tqdm

import argparse
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw
from helper import save_model, load_model, plot_curve

# Define the diffusion process (inspired by https://github.com/dome272/Diffusion-Models-pytorch (https://www.youtube.com/watch?v=TBCRlnwJtZU) and based on the DDPM paper )
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, data_size=64, device="cuda"):
        # Set beta, alpha and alpha_hat (cumulative)
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.data_size = data_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # Returns a list of cummlative products of alpha[t]
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # Returns a list with evenly spaced values of beta (variance) between start and end
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # Return a list of N random timesteps (N is number of data intences)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # Takes data (array) and adds noise according to the timesteps (array)
    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None]
        epsilon = torch.randn_like(x)
        return    sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    # Takes an example x and adds t steps of noise to make x_t, than reconstrct x from x_t
    def reconstruct(self, model, x, t, progress_bar=True):
        model.eval()
        with torch.no_grad():
            #x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(range(1, t), position=0, disable=not progress_bar):
                # Instead of using reversed(range(1, t)) in tqdm
                j = t - i
                # Make a list of t for each example in x
                ts = (torch.ones(x.shape[0]) * j).int().to(self.device)
                predicted_noise = model(x, ts)
                alpha = self.alpha[ts][:, None]
                alpha_hat = self.alpha_hat[ts][:, None]
                beta = self.beta[ts][:, None]
                # We don't need noise for the last iteration
                if j > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # We use the predicted noise as the mean of the denoising noise we add to the data (DDPM paper Algorithm 2)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            model.train()
            #x = (x.clamp(-1, 1) + 1) / 2
            #x = (x * 255).type(torch.uint8)
            return x
        
# The neural network that estimates the added noise
class MLP(nn.Module):
    def __init__(self, data_dim=196, hidden_dim=512, emb_dim=256, device='cuda'):
        super(MLP, self).__init__()

        self.emb_dim = emb_dim
        self.device = device

        self.l1 = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l6 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l7 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l9 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l10 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l11 = nn.Linear(hidden_dim, data_dim)

        # Project the encoded timestep into the hidden layer dimension (to add it later)
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, hidden_dim))


    # Positional timestep embedding (encode the timestep with sin/cos)
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        # Encode the timestep
        t = self.pos_encoding(t, self.emb_dim)
        # Project the timestep into the hidden layer dimension
        t = self.emb_layer(t)
        # Add the encoded+projected timestep to each hidden layer output
        # (to give the neural network the information about the timestep)
        a1 = self.l1(x) + t
        a2 = self.l2(a1) + t
        a3 = self.l3(a2) + t
        a4 = self.l4(a3) + t
        a5 = self.l5(a4) + t
        a6 = self.l6(a5) + t
        a7 = self.l7(a6) + t
        a8 = self.l8(a7) + t
        a9 = self.l9(a8) + t
        a10 = self.l10(a9) + t
        return self.l11(a10)
    

def train(model, diffusion, loss_fn, optimizer, x_train, epochs=1000, device='cuda', progress_plot=True, log_name="", train_loss=[]):
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        t = diffusion.sample_timesteps(x_train.shape[0]).to(device)

        x_t, noise = diffusion.noise_data(x_train, t)
        predicted_noise = model(x_t, t)
        loss = loss_fn(noise, predicted_noise)

        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(MSE=train_loss[-1])
        if progress_plot:
            if (epoch+1)%1000 == 0:
                plot_curve('progress_'+log_name, blue=train_loss)
                save_model('progress_'+log_name, model, train_loss)

    return train_loss




def main(noise_steps, lr, epochs, device, hidden_dim, beta_start=1e-4, beta_end=0.02, retrain=False):
    if device=='none':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, _, _, _, = preprocess_unsw()
    # Convert the data to PyTorch Tensor in the GPU
    #x_train, y_train = torch.Tensor(x_train).to(device), torch.Tensor(y_train).long().to(device)
    #x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).long().to(device)
    x_train = torch.Tensor(x_train).to(device)

    model = MLP(data_dim=196, hidden_dim=hidden_dim, emb_dim=256, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    process = Diffusion(data_size=196, noise_steps=noise_steps, beta_start=beta_start, beta_end=beta_end, device=device)

    log_name = "DIFFUSION"+"_T_"+str(noise_steps)+"_B_"+str(beta_start)+"_"+str(beta_end)+"_"+str(loss)[:-2]+"_LR_"+str(lr)+"_E_"+str(epochs)+"_H_"+str(10)+"-"+str(hidden_dim)+"_"+device

    train_loss=[]
    
    if retrain:
        print("Retraining "+log_name)
        train_loss = load_model(log_name, model)
    else :
        print("Training "+log_name)

    train_loss = train(model=model, diffusion=process, loss_fn=loss, optimizer=optimizer, x_train=x_train, epochs=epochs, device=device, log_name=log_name, train_loss=train_loss)




    save_model(log_name, model, train_loss)
    plot_curve(log_name, train_loss)

if __name__ == "__main__":

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", required=True, default="none", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("-e", "--epochs", required=True, default=100, type=int, help="Number of epochs to train the model.")
    parser.add_argument("-l", "--learning_rate", required=True, default=1e-3, type=float, help="Learning rate")
    parser.add_argument("-n", "--noise_steps", required=True, default=1000, type=int, help="Noise steps")
    parser.add_argument("-i", "--hidden_dim", required=True, default=1024, type=int, help="Dimension of hidden layer.")
    parser.add_argument("-r", "--retrain", required=False, default=0, type=int, help="Load and retrain model.")
    parser.add_argument("-bs", "--beta_start", required=True, default=1e-4, type=float, help="Start value for beta")
    parser.add_argument("-be", "--beta_end", required=True, default=0.02, type=float, help="Start value for beta")
    args = parser.parse_args()

    main(hidden_dim=args.hidden_dim, noise_steps=args.noise_steps, lr=args.learning_rate, epochs=args.epochs, device=args.device, retrain=args.retrain, beta_start=args.beta_start, beta_end=args.beta_end)