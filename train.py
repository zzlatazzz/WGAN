import torch

import os
from tqdm import tqdm

from losses import generator_loss, discriminator_loss, l1_loss
from models import Generator, Discriminator
from configs import TrainConfig, GeneratorConfig, DiscriminatorConfig
from test_log import test_log
from wandbwriter import WanDBWriter
from data import get_data

from copy import deepcopy

from functools import reduce

def save_checkpoint(path, generator, discriminator):
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()}, path)
    
def load_checkpoint(path, generator, discriminator):
    full_ckpt = torch.load(path)
    generator.load_state_dict(full_ckpt['generator'])
    discriminator.load_state_dict(full_ckpt['discriminator'])
    return generator, discriminator


def visualize_and_log_image(image, max_val_to_visualize, logger, name):
    
    image = (image / max_val_to_visualize).clip(0, 1)
    image[0, 0] = 0
    image[0, 1] = 1
    logger.add_image(name, image)


def train():
    
    train_loader, test_loader, momentums_transform, points_transform = get_data()

    config = TrainConfig()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device)

    generator = Generator(GeneratorConfig()).to(device)
    discriminator = Discriminator(DiscriminatorConfig()).to(device)

    opt_g = torch.optim.Adam(list(generator.parameters()), lr=config.generator_lr, betas=config.generator_betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=config.discriminator_betas)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=config.generator_gamma)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=config.discriminator_gamma)

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    
    for i in range(len(generator.consts)):
        generator.consts[i].weight.data = torch.tensor([[config.start_const]]).to(device)
    
    current_step = (config.last_epoch + 1) * len(train_loader)
    
    logger = WanDBWriter(config.project_name, config.wandb_api_key)
    tqdm_bar = tqdm(total=(config.num_epochs - config.last_epoch - 1) * len(train_loader))
    
    alpha = config.alpha
    r, dm = 0, 0
    beta = config.beta
    
    logger.add_scalar('alpha', alpha)
    logger.add_scalar('beta', beta)
    
    for epoch in range(config.last_epoch + 1, config.num_epochs):
        loss_gen = 0
        loss_l1 = 0
        scheduler_g.step()
        scheduler_d.step()
        for idx, batch in enumerate(train_loader):
            if idx <= config.num_first_disc_steps or idx % config.num_steps_in_cycle:
                generator.eval()
                discriminator.train()
                current_step += 1
                logger.set_step(current_step, '')
            
                tqdm_bar.update(1)
                ims, moms, ps = batch
                ims, moms, ps = ims.to(device), moms.to(device), ps.to(device)
                z = torch.randn(len(ims), config.noise_dim).type_as(ims)
            
                fake_ims = generator(z, moms, ps)
                
                logger.add_scalar('fake_max_val', fake_ims.max())
                logger.add_scalar('LR', torch.tensor(scheduler_d.get_last_lr()[-1]))
                
                discriminator.update_const(0.9 ** r)
                
                d1 = discriminator(ims, moms, ps)
                d2 = discriminator(fake_ims.detach(), moms, ps)
                
                dist = max(d1) - min(d2)
                dm = alpha * dm + (1 - alpha) * dist
                clbr_dm = dm / beta
                r = max(0, clbr_dm.detach() / (1 - clbr_dm.detach()))
                logger.add_scalar('r', r)
                logger.add_scalar('discriminator_const', discriminator.const)
                
                loss_disc = 0
                loss_disc = discriminator_loss(d1, d2, config.task)
            
                loss_disc.backward()
                
            
                d_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).cpu() for p in discriminator.parameters() if p.grad is not None]), 2,).item()
                logger.add_scalar('discriminator_grad_norm', d_grad_norm)                
                
                opt_d.step()
                opt_d.zero_grad()
        
            if idx > config.num_first_disc_steps and not idx % config.num_steps_in_cycle:
                generator.train()
                discriminator.eval()
                current_step += 1
                logger.set_step(current_step, '')
            
                tqdm_bar.update(1)
                ims, moms, ps = batch
                ims, moms, ps = ims.to(device), moms.to(device), ps.to(device)
                z = torch.randn(len(ims), config.noise_dim).type_as(ims)
                    
                fake_ims = generator(z, moms, ps)
                logger.add_scalar('fake_max_val', fake_ims.max())
                logger.add_scalar('LR', torch.tensor(scheduler_d.get_last_lr()[-1]))
            
                loss_gen = generator_loss(discriminator(fake_ims, moms, ps)) + config.reg_coef * reduce(lambda x, y: x*y, [generator.consts[i].weight for i in range(len(generator.consts))])
                
                loss_l1 = l1_loss(ims, fake_ims)
            
                loss_gen.backward()
                g_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).cpu() for p in generator.parameters() if p.grad is not None]), 2,).item()
                logger.add_scalar('generator_grad_norm', g_grad_norm)
            
                opt_g.step()
                opt_g.zero_grad()
                
                for i in range(len(generator.consts)):
                    logger.add_scalar('generator_const_' + str(i), generator.consts[i].weight.item())
            
            visualize_and_log_image(fake_ims[0], config.max_val_to_visualize_log, logger, 'fake_log_image')
            visualize_and_log_image(ims[0], config.max_val_to_visualize_log, logger, 'real_log_image')
        
            if current_step > 0 and current_step % config.num_steps_to_log_losses == 0:
                logger.add_scalar('discriminator_loss_train', loss_disc)
                logger.add_scalar('generator_loss_train', loss_gen)
                logger.add_scalar('l1_loss_train', loss_l1)
                logger.add_scalar('epoch', epoch)
                
                
                test_log(generator, discriminator, test_loader, logger, current_step, config, momentums_transform, points_transform)
            
            
            if current_step % config.num_steps_to_save_checkpoint == 0 and current_step != 0:
                save_checkpoint(config.checkpoint_path + str(current_step), generator, discriminator)
        
        save_checkpoint(config.checkpoint_path, generator, discriminator)
        
if __name__ == "__main__":
    train()
