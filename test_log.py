import torch
from IPython.display import clear_output
from PIL import Image
import numpy as np

from losses import generator_loss, discriminator_loss, l1_loss
from prd_score import compute_auc_pr, compute_prd_from_embedding, plot
from functions import get_images, plot_hist
from sklearn.cluster import KMeans

def inverse_image_transform(images):
    return 10 ** images - 1


def log_image(image, norm, logger, name):
    
    image = (image / norm).clip(0, 1)
    image[0, 0] = 0
    image[0, 1] = 1
    logger.add_image(name, image)


def test_log(generator, discriminator, test_loader, logger, current_step, config, momentums_transform, points_transform):

    device = torch.device(config.device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        gen_loss, disc_loss, loss_l1 = 0, 0, 0
        real_log_ims, fake_log_ims, momss, pss = [], [], [], []
        real_disc_outputs, fake_disc_outputs = [], []
        real_nonzero_points, fake_nonzero_points = [], []
        for idx, batch in enumerate(test_loader):
            logger.set_step(current_step, '')
            
            ims, moms, ps = batch
            real_log_ims.append(ims)
            momss.append(moms)
            pss.append(ps)
            ims, moms, ps = ims.to(device), moms.to(device), ps.to(device)
            z = torch.randn(len(ims), config.noise_dim).to(device)
            z = z.type_as(ims)
        
            fake_ims = generator(z, moms, ps)
            fake_log_ims.append(fake_ims.detach())
            
            real_nonzero_points.extend((ims != 0).sum(dim=(1, 2, 3)).flatten().tolist())
            fake_nonzero_points.extend((fake_ims != 0).sum(dim=(1, 2, 3)).flatten().tolist())

            d_real = discriminator(ims, moms, ps)
            d_fake = discriminator(fake_ims, moms, ps)
                              
            real_disc_outputs.extend(d_real.flatten().tolist())
            fake_disc_outputs.extend(d_fake.flatten().tolist())
                    
            disc_loss += discriminator_loss(d_real, d_fake.detach(), config.task)
            gen_loss += generator_loss(d_fake)
            loss_l1 += l1_loss(ims, fake_ims)
                                      
        logger.add_scalar('discriminator_loss_test', disc_loss / len(test_loader))
        logger.add_scalar('generator_loss_test', gen_loss / len(test_loader))
        logger.add_scalar('l1_loss_test', loss_l1 / len(test_loader))
            
                
        fake_log_ims = torch.cat(fake_log_ims, dim=0).detach().cpu()
        real_log_ims = torch.cat(real_log_ims, dim=0).detach().cpu()
        fake_ims = inverse_image_transform(fake_log_ims)
        real_ims = inverse_image_transform(real_log_ims)
        
        if current_step % config.num_steps_to_log_prd == 0:
            logger.add_scalar('prd_score_test', compute_auc_pr(*compute_prd_from_embedding(fake_ims.view(len(fake_ims), -1), real_ims.view(len(real_ims), -1), num_clusters=config.num_clusters, num_runs=config.num_runs)))
        
        log_image(torch.cat(list(real_ims[:config.num_test_images_to_log]),dim=2), config.max_val_to_visualize, logger, 'real_images')
        log_image(torch.cat(list(fake_ims[:config.num_test_images_to_log]),dim=2), config.max_val_to_visualize, logger, 'fake_images')

        log_image(torch.cat(list(real_log_ims[:config.num_test_images_to_log]),dim=2), config.max_val_to_visualize_log, logger, 'real_log_images')
        log_image(torch.cat(list(fake_log_ims[:config.num_test_images_to_log]),dim=2), config.max_val_to_visualize_log, logger, 'fake_log_images')
                
        if current_step % config.num_steps_to_log_funcs == 0:
            
            momss = torch.cat(momss, dim=0).detach().cpu()
            pss = torch.cat(pss, dim=0).detach().cpu()
            
            kmeans_preds = torch.tensor(KMeans(n_clusters=3).fit(np.array([[0.48056529, 0.52330842, 0.08993791, 0.50162217, 0.61243354],
                                                                           [0.48014588, 0.53682416, 0.6204677 , 0.50254773, 0.5247629 ],
                                                                           [0.47953647, 0.55149798, 0.09223902, 0.50615411, 0.43817334]])
                                                                ).predict(torch.cat([momss, pss], dim=1)))
            
            momss = momentums_transform.inverse_transform(momss)
            pss = points_transform.inverse_transform(pss)
            
            unique = torch.unique(torch.tensor(kmeans_preds))
            set1 = real_ims[kmeans_preds == 0], fake_ims[kmeans_preds == 0], momss[kmeans_preds == 0], pss[kmeans_preds == 0]
            set2 = real_ims[kmeans_preds == 1], fake_ims[kmeans_preds == 1], momss[kmeans_preds == 1], pss[kmeans_preds == 1]
            set3 = real_ims[kmeans_preds == 2], fake_ims[kmeans_preds == 2], momss[kmeans_preds == 2], pss[kmeans_preds == 2]
            
            sets = [set1, set2, set3]
            mean_prd_phys = 0
            mean_prd_ims = 0
            for i, (real_ims, fake_ims, momss, pss) in enumerate(sets):
                
                im1, im2, im3, im4, im5, total_real, total_fake = get_images(real_ims, momss, pss, fake_ims, momss, pss, config)
                im6 = plot_hist(torch.tensor(real_disc_outputs), torch.tensor(fake_disc_outputs), config, 100, None, 'Discriminator output')
                im7 = plot_hist(torch.tensor(real_nonzero_points), torch.tensor(fake_nonzero_points), config, 100, None, 'Number of nonzero points')
                im8 = plot([compute_prd_from_embedding(fake_ims.view(len(fake_ims), -1), real_ims.view(len(real_ims), -1), num_clusters=config.num_clusters, num_runs=config.num_runs)], title='prd for images')
                
                c = compute_prd_from_embedding(torch.cat(total_fake).view(len(torch.cat(total_fake)), -1), torch.cat(total_real).view(len(torch.cat(total_real)), -1), num_clusters=config.phys_num_clusters, num_runs=config.phys_num_runs)
                cap = compute_auc_pr(*c)
                mean_prd_phys += cap * len(real_ims)
                logger.add_scalar('prd_score_phys'+ str(i), cap)
                im9 = plot([c], title='prd for phys values')
                
                c = compute_prd_from_embedding(fake_ims.view(len(fake_ims), -1), real_ims.view(len(real_ims), -1), num_clusters=config.num_clusters, num_runs=config.num_runs)
                cap = compute_auc_pr(*c)
                mean_prd_ims += cap * len(real_ims)
                logger.add_scalar('prd_score_test'+ str(i), cap)
                
                for j, im in enumerate([im1, im2, im3, im4, im5, im6, im7, im8, im9]):
                    logger.add_image('set' + str(i) + '_' + str(j + 1), im)
                    
            logger.add_scalar('mean_prd_score_phys', mean_prd_phys / (len(set1[0]) + len(set2[0]) + len(set3[0])))
            logger.add_scalar('mean_prd_score_test', mean_prd_ims / (len(set1[0]) + len(set2[0]) + len(set3[0])))
                
            clear_output()
