import torch
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm
import numpy as np

import pickle
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from data import preprocess_unsw
from helper import save_model, accuracy

def fgsm(model, loss, optimizer, epsilon, epsilon_steps, x_test, y_test, log_name):
    test_adv_acc = []
    epsilon_index = []

    if x_test.get_device()==-1:
        device = 'cpu'
    else:
        device = 'cuda'

    # Convert the PyTorch model into an ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(196),
        nb_classes=2,
    )

    # Test the robustness of the model for different values of epsilon (perturbation amplitude)
    pbar = tqdm([epsilon*i for i in range(epsilon_steps)])
    for eps in pbar:
        attack = FastGradientMethod(estimator=classifier,
                                    eps=eps,
                                    batch_size=1024)
        # Generate adversarial with FGSM on the testing set (ART works on the CPU)
        x_test_adv = attack.generate(x=x_test.cpu().numpy(), y=y_test.cpu().numpy())
        # Test the model on adversarial data
        with torch.no_grad():
            x_test_adv = torch.Tensor(x_test_adv).to(device)
            pred = model(x_test_adv)
            test_adv_acc.append(accuracy(pred, y_test))
            epsilon_index.append(epsilon)

        pbar.set_postfix(ADV_ACC=test_adv_acc[-1])
    
    with open('results/ADV_'+log_name+'.np', 'wb') as file:
        np.save(file, x_test_adv.cpu().numpy())
    
    with open('results/ADV_'+log_name+'.logs', 'wb') as file:
        pickle.dump(test_adv_acc, file)

    return x_test_adv