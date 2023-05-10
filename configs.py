from dataclasses import dataclass
import torch

@dataclass
class GeneratorConfig:
    noise_dim = 64
    momentums_dim = 3
    points_dim = 2

    emb_dim = 64
    
    first_shape = (3, 3)
    second_shape = (7, 7)
    last_shape = (30, 30)

    num_channels = [16, 32, 64, 128]

    kernel_size = 3
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DiscriminatorConfig:
    momentums_dim = 3
    points_dim = 2

    emb_dim = 64
    
    num_channels = [16, 16, 32, 64, 64]
    strides = [2, 2, 2, 2, 2]
    paddings = [2, 1, 1, 1, 1]

    kernel_size = 3


@dataclass
class TrainConfig:
    batch_size = 128
    batch_size_test = 1024

    generator_lr = 2e-4
    discriminator_lr = 2e-4

    generator_betas = (0.5, 0.999)
    discriminator_betas = (0.5, 0.999)

    generator_gamma = 0.98
    discriminator_gamma = 0.98
    
    beta = 1.5
    alpha = 0.9999
    
    start_const = 1.
    
    reg_coef = 1e-5
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    noise_dim = 64

    seed = 6
    last_epoch = -1
    num_epochs = 155
    num_first_disc_steps = 0
    num_steps_in_cycle = 6
    
    first_const_step = 100
    
    num_runs = 20
    num_clusters = 60

    phys_num_runs = 30
    phys_num_clusters = 400

    checkpoint_dir = 'checkpoints'
    checkpoint_path = 'checkpoints/checkpoint.s'
    
    checkpoint_start = 'checkpoint.start'

    data_path = 'data.npz'

    num_steps_to_log_losses = 1000
    num_steps_to_log_prd = 10000
    num_steps_to_log_funcs = 10000

    num_test_images_to_log = 8

    num_steps_to_save_checkpoint = 20000

    max_val_to_visualize_log = 2.
    max_val_to_visualize = 100.

    project_name = 'WGAN'
    wandb_api_key = 'e750a8f4817e7ec797289407d10908d46e901ab8'

    task = 'HINGE' #'WASSERSTEIN'
