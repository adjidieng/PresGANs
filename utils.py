import os 
import matplotlib.pyplot as plt

def plot_kde(samples, epoch, name, title='', cmap='Blues', save_path=None):
    samples = samples.cpu().numpy()
    sns.set(font_scale=2)
    f, ax = plt.subplots(figsize=(4, 4))
    sns.kdeplot(samples[:, 0], samples[:,1], cmap=cmap, ax=ax, n_levels=20, shade=True)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axis('off')
    plt.title(title)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, '{}_{}.pdf'.format(name, epoch)))
    plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
