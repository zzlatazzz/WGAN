import torch

def discriminator_loss(y, y_hat, task):
    if task == 'HINGE':
        return torch.mean(torch.nn.functional.relu(1 - y) + torch.nn.functional.relu(1 + y_hat))
    elif task == 'WASSERSTEIN':
        return torch.mean(y_hat) - torch.mean(y)

def generator_loss(y_hat):
    return - torch.mean(y_hat)

def l1_loss(real_img, fake_img):
    return torch.nn.L1Loss()(fake_img, real_img)


