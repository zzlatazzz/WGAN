import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Sequential( spectral_norm( nn.Linear( config.momentums_dim + config.points_dim, config.emb_dim )),
                                  nn.LeakyReLU(),
                                  spectral_norm( nn.Linear( config.emb_dim, config.emb_dim )))
        
        self.model = nn.ModuleList([
            spectral_norm( nn.Linear( config.emb_dim, config.num_channels[0] * np.prod(config.first_shape) )), # 64 -> 144
            nn.Sequential(nn.Unflatten( 1, (config.num_channels[0], *config.first_shape) ),
            nn.LeakyReLU(),
            nn.Upsample(config.second_shape)), # 16, 7, 7
            
            spectral_norm( nn.Conv2d( config.num_channels[0], config.num_channels[1], config.kernel_size, padding='same' )), # 32, 7, 7
            nn.Sequential( nn.BatchNorm2d( config.num_channels[1] ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)), # 32, 14, 14
            
            spectral_norm( nn.Conv2d( config.num_channels[1], config.num_channels[2], config.kernel_size, padding='same' )),
            nn.Sequential(nn.BatchNorm2d( config.num_channels[2] ),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)), # 64, 28, 28
            
            spectral_norm( nn.Conv2d( config.num_channels[2], config.num_channels[3], config.kernel_size, padding='same' )),
            nn.Sequential(nn.BatchNorm2d( config.num_channels[3] ),
            nn.LeakyReLU(),
            nn.Upsample(config.last_shape)), # 128, 30, 30
            
            spectral_norm( nn.Conv2d( config.num_channels[3], 1, config.kernel_size, padding='same' )),
            nn.ReLU()
        ])
        
        self.consts = nn.ModuleList([nn.Linear(1, 1, bias=False) for i in range(6)])
        
    
    def forward(self, noise, momentum, point):
        emb = self.consts[0](self.emb(torch.cat([momentum, point], dim=1))[:, :, None])[:, :, 0]
        img = self.consts[1]((self.model[0](noise * emb))[:, :, None])[:, :, 0]
        
        for i in range(1, len(self.model)):
            img = self.model[i](img)
            if not i % 2:
                img = (self.consts[i // 2 + 1](img[:, :, :, :, None]))[:, :, :, :, 0]
                
        return img



class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.ModuleList([
            spectral_norm( nn.Linear( config.momentums_dim + config.points_dim, config.emb_dim )),
            spectral_norm( nn.Linear( config.emb_dim, config.emb_dim )),
        ])

        self.conv_net = nn.ModuleList([
            spectral_norm(nn.Conv2d( 1, config.num_channels[0], config.kernel_size, stride=config.strides[0], padding=config.paddings[0] )),
            spectral_norm(nn.Conv2d( config.num_channels[0], config.num_channels[1], config.kernel_size, stride=config.strides[1], padding=config.paddings[1] )),
            spectral_norm(nn.Conv2d( config.num_channels[1], config.num_channels[2], config.kernel_size, stride=config.strides[2], padding=config.paddings[2] )),
            spectral_norm(nn.Conv2d( config.num_channels[2], config.num_channels[3], config.kernel_size, stride=config.strides[3], padding=config.paddings[3] )),
            spectral_norm(nn.Conv2d( config.num_channels[3], config.num_channels[4], config.kernel_size, stride=config.strides[4], padding=config.paddings[4] ))
        ])

        self.fc1 = spectral_norm(nn.Linear(config.emb_dim, 1))
        
        self.const = 1.

    def update_const(self, const):
        self.const = const
        
    def forward(self, img, momentum, point):
        for layer in self.conv_net:
            img = nn.LeakyReLU()(layer(img) * self.const)
        img = nn.Flatten()(img)
        
        emb = self.emb[1](nn.LeakyReLU()(self.emb[0](torch.cat([momentum, point], dim=1)) * self.const)) * self.const
        res = self.fc1(img * emb) * self.const
        return res