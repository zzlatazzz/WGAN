import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import io
from scipy.interpolate import interp2d

from prd_score import compute_auc_pr, compute_prd_from_embedding, plot

def get_assymetry(data, ps, points, orthog=False):
    first = True
    assym_res = []
    
    x = torch.linspace(-14.5, 14.5, 30)
    y = torch.linspace(-14.5, 14.5, 30)
    xx, yy = torch.meshgrid(x, y)
        
    for i in range(len(data)):
        img = data[i]
        p = ps[i]
        point = points[i, :2]
        zoff = 25
        point0 = point[0] + zoff*p[0]/p[2]
        point1 = point[1] + zoff*p[1]/p[2]
    
        if orthog:
            line_func = lambda x: (x - point0) / p[0] * p[1] + point1
        else:
            line_func = lambda x: -(x - point0) / p[1] * p[0] + point1
    
        idx = torch.where(yy - line_func(xx) < 0)
        if (not orthog and p[1]<0):
            idx = torch.where(yy - line_func(xx) > 0)
    
        zz = torch.ones((30, 30))
        zz[idx] = 0
    
        assym = (torch.sum(img * zz) - torch.sum(img * (1 - zz))) / torch.sum(img)
        assym_res.append(assym.item())
    return torch.tensor(assym_res)
    
    
def get_shower_width(data, ps, points, orthog=False):
    res, spreads = [], []
    for i in range(min(len(data), 10000)):
        img = data[i]
        p = ps[i]
        point = points[i]
        zoff = 25
        point0 = point[0] + zoff*p[0]/p[2]
        point1 = point[1] + zoff*p[1]/p[2]
        if orthog:
            line_func = lambda x: -(x - point0) / p[0] * p[1] + point1
        else:
            line_func = lambda x:  (x - point0) / p[1] * p[0] + point1
    
        x = torch.linspace(-14.5, 14.5, 30)
        y = torch.linspace(-14.5, 14.5, 30)

        bb = interp2d(x, y, img, kind='cubic')

        x_ = torch.linspace(-14.5, 14.5, 100)
        y_ = line_func(x_)
        rescale = torch.sqrt(1+(p[1]/p[0])*(p[1]/p[0])).item()

        sum0, sum1, sum2 = 0, 0, 0
        for i in range(100):
            ww = bb(x_[i], y_[i]).item(0)
            if ww < 0: ww = 0
            sum0 += ww
            sum1 += rescale*x_[i].item()*ww
            sum2 += (rescale*x_[i].item())*(rescale*x_[i].item())*ww
        sum1 = sum1/max(sum0, 1e-16)
        sum2 = sum2/max(sum0, 1e-16)
        if sum2 >= sum1*sum1 :
            sigma = (sum2 - sum1*sum1) ** 0.5
            spreads.append(sigma)
        else:
            spreads.append(0)
    return torch.tensor(spreads)

    
def get_ms_ratio2(data, ps, i, alpha=0.1):
    img = data[i]
    ms = torch.sum(img)
    ms_ = ms * alpha
    num = torch.sum((img >= ms_))
    return num / 900.


def plot_hist(real, fake, config, bins, range_, xlabel, ylabel=None):
    matplotlib.rcParams.update({'font.size': 14})
    plt.hist(real, bins=bins, range=range_, color='red', alpha=0.3, density=True, label='Real')
    plt.hist(fake, bins=bins, range=range_, color='blue', alpha=0.3, density=True, label='Fake')
    
    real[torch.logical_not(torch.isfinite(real))] = 1e6
    fake[torch.logical_not(torch.isfinite(fake))] = 1e6
    plt.title('prd = ' + str(round(compute_auc_pr(*compute_prd_from_embedding(fake.view(len(fake), -1), real.view(len(real), -1), num_clusters=config.phys_num_clusters, num_runs=config.phys_num_runs)), 3)))
    plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend(loc='best')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, dpi=300, bbox_inches='tight', pad_inches=1.5)
    plt.close()
    im = Image.open(img_buf)
    return im


def get_images(real_imgs, real_p, real_point, fake_imgs, fake_p, fake_point, config):

    total_real, total_fake = [], []
    
    assymetry_direct_real = get_assymetry(real_imgs, real_p, real_point)
    assymetry_perp_real = get_assymetry(real_imgs, real_p, real_point, orthog=True)
    assymetry_direct_fake = get_assymetry(fake_imgs, fake_p, fake_point)
    assymetry_perp_fake = get_assymetry(fake_imgs, fake_p, fake_point, orthog=True)

    im1 = plot_hist(assymetry_direct_real, assymetry_direct_fake, config, 100, [-1, 1], 'Longitudual cluster asymmetry')
    im2 = plot_hist(assymetry_perp_real, assymetry_perp_fake, config, 100, [-1, 1], 'Transverse cluster asymmetry')
    
    total_real.append(assymetry_direct_real)
    total_real.append(assymetry_perp_real)
    total_fake.append(assymetry_direct_fake)
    total_fake.append(assymetry_perp_fake)
    
    shower_width_real_direct = get_shower_width(real_imgs, real_p, real_point)
    shower_width_real_perp = get_shower_width(real_imgs, real_p, real_point, orthog=True)
    shower_width_fake_direct = get_shower_width(fake_imgs, fake_p, fake_point)
    shower_width_fake_perp = get_shower_width(fake_imgs, fake_p, fake_point, orthog=True)

    im3 = plot_hist(shower_width_real_direct, shower_width_fake_direct, config, 50, [0, 15], 'Cluster longitudual width [cm]', 'Arbitrary units')
    im4 = plot_hist(shower_width_real_perp, shower_width_fake_perp, config, 50, [0, 10], 'Cluster trasverse width [cm]', 'Arbitrary units')
    
    total_real.append(shower_width_real_direct)
    total_real.append(shower_width_real_perp)
    total_fake.append(shower_width_fake_direct)
    total_fake.append(shower_width_fake_perp)
    
    alpha = torch.linspace(-5, 0, 50)
    sparsity_real = []
    sparsity_fake = []

    for i in range(min(len(real_imgs), 3000)):
        v_r = []
        v_f = []
        for alpha_ in alpha:
            v_r.append(get_ms_ratio2(real_imgs, real_p, i, pow(10,alpha_)))
            v_f.append(get_ms_ratio2(fake_imgs, fake_p, i, pow(10,alpha_)))
        
        sparsity_real.append(v_r)
        sparsity_fake.append(v_f)

    res_r = torch.tensor(sparsity_real)
    res_f = torch.tensor(sparsity_fake)

    means_r = torch.mean(res_r, axis=0)
    stddev_r = torch.std(res_r, axis=0)

    means_f = torch.mean(res_f, axis=0)
    stddev_f = torch.std(res_f, axis=0)

    matplotlib.rcParams.update({'font.size': 14})
    plt.plot(alpha, means_r, color='red')
    plt.fill_between(alpha, means_r-stddev_r, means_r+stddev_r, color='red', alpha=0.3)
    plt.plot(alpha, means_f, color='blue')
    plt.fill_between(alpha, means_f-stddev_f, means_f+stddev_f, color='blue', alpha=0.3)
    
    plt.legend(['Real', 'Fake'])
    plt.title('Sparsity')
    plt.xlabel('log10(Threshold/GeV)')
    plt.ylabel('Fraction of cells above threshold')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, dpi=300, bbox_inches='tight', pad_inches=1.5)
    plt.close()
    im5 = Image.open(img_buf)
    return im1, im2, im3, im4, im5, total_real, total_fake
    


