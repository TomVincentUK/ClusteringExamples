"""
Functions for data clustering demonstrations
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def Gaussian(x, mu, sigma):
    """Simple normal distribution with height 1"""
    return np.exp(-0.5*((x-mu)/sigma)**2)

def gen_domains(populations, means, stds, image_size=128, feature_size=10):
    """
    Generate a fake image composed of 'materials' with an associated
    mean and standard deviation.
    """
    populations = np.asarray(populations)
    means = np.asarray(means)
    stds = np.asarray(stds)
    
    # Generate smooth noise
    field = gaussian_filter(
        np.random.randn(image_size, image_size),
        feature_size
    )
    
    # Convert relative populations to percentage of total image
    bounds = np.zeros(populations.size+1)
    bounds[1:] = np.cumsum(100*populations/populations.sum())
    materials = [
        (field>=np.percentile(field, bounds[i]))
        & (field<=np.percentile(field, np.clip(bounds[i+1], 0, 100)))
        for i in range(populations.size)
    ]
    
    image = np.sum([
        (mean+np.random.randn(image_size, image_size)*std)*mat.astype(float)
        for mean, std, mat in zip(means, stds, materials)
    ], axis=0)
    
    return image, materials

def gen_domains2D(
        populations,
        means,
        stds,
        pearsons,
        image_size=128,
        feature_size=10):
    """
    Generate a pair of fake images composed of 'materials' with an
    associated 2D mean and covariance.
    """
    populations = np.asarray(populations)
    means = np.asarray(means)
    stds = np.asarray(stds)
    pearsons = np.asarray(pearsons)
    
    # Generate smooth noise
    field = gaussian_filter(
        np.random.randn(image_size, image_size),
        feature_size
    )
    
    # Convert relative populations to percentage of total image
    bounds = np.zeros(populations.size+1)
    bounds[1:] = np.cumsum(100*populations/populations.sum())
    materials = [
        (field>=np.percentile(field, bounds[i]))
        & (field<=np.percentile(field, np.clip(bounds[i+1], 0, 100)))
        for i in range(populations.size)
    ]
    
    # mean zero, identity covariance noise to be modified
    identity_noise = np.random.randn(2, image_size, image_size)
    
    images = np.zeros_like(identity_noise)
    for mat, mean, std, pearson in zip(materials, means, stds, pearsons):
        M = np.array([
            [(1+pearson)*std[0], (pearson-1)*std[0]],
            [(1+pearson)*std[1], (1-pearson)*std[1]]
        ]) * np.sqrt(2)
        
        images += (
            mean
            + (M@identity_noise.reshape(2, -1)).T
            ).T.reshape(images.shape) * mat.astype(float)
    
    return images, materials

def plot_domains(image, materials, width=12):
    n_mat = len(materials)
    
    fig = plt.figure(figsize=(width, width/2 + width/(2*n_mat)))
    gs = plt.GridSpec(
        figure=fig,
        nrows=2,
        ncols=n_mat+1,
        width_ratios=n_mat*[1/n_mat,] + [1,]
    )
    
    im_ax = fig.add_subplot(gs[0, :n_mat])
    im = im_ax.imshow(image, cmap=plt.cm.Spectral)
    im_ax.set_title(f'image')
    cbar = fig.colorbar(im, ax=im_ax)
    
    material_axes = [fig.add_subplot(gs[1, i]) for i in range(n_mat)]
    for i, (ax, mat) in enumerate(zip(material_axes, materials)):
        ax.imshow(mat, interpolation='nearest')
        ax.set_title(f'material {i+1}')
    
    hist_ax = fig.add_subplot(gs[:, -1])
    hist_ax.hist(image.flatten(), bins=int(np.sqrt(image.size)))
    hist_ax.scatter(
        image.flatten(),
        0*image.flatten(),
        marker='|',
        lw=1,
        alpha=0.5,
        s=500,
        c='k',
        zorder=3
    )
    hist_ax.set(
        title=f'values',
        yticks=[]
    )
    for side in ('top', 'left', 'right'):
        hist_ax.spines[side].set_visible(False)

    for ax in material_axes+[im_ax,]:
        ax.set_axis_off()
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig

def plot_domains2D(images, materials, width=12):
    n_mat = len(materials)
    
    fig = plt.figure(figsize=(width, width/4 + width/(2*n_mat)))
    gs = plt.GridSpec(
        figure=fig,
        nrows=2,
        ncols=n_mat+1,
        width_ratios=n_mat*[1/n_mat,] + [1,]
    )
    
    subgrid0 = gs[0, :n_mat].subgridspec(1, 2)
    
    im_ax0 = fig.add_subplot(subgrid0[0])
    im0 = im_ax0.imshow(images[0], cmap=plt.cm.Spectral)
    im_ax0.set_title(f'image $A$')
    cbar0 = fig.colorbar(im0, ax=im_ax0)
    
    im_ax1 = fig.add_subplot(subgrid0[1])
    im1 = im_ax1.imshow(images[1], cmap=plt.cm.Spectral)
    im_ax1.set_title(f'image $B$')
    cbar1 = fig.colorbar(im1, ax=im_ax1)
    
    material_axes = [fig.add_subplot(gs[1, i]) for i in range(n_mat)]
    for i, (ax, mat) in enumerate(zip(material_axes, materials)):
        ax.imshow(mat, interpolation='nearest')
        ax.set_title(f'material {i+1}')
    
    subgrid1 = GridSpecFromSubplotSpec(
        nrows=2,
        ncols=2,
        subplot_spec=gs[:, -1],
        height_ratios=[0.25, 1],
        width_ratios=[1, 0.25]
    )
    scatter_ax = fig.add_subplot(subgrid1[1, 0])
    hist_x_ax = fig.add_subplot(subgrid1[0, 0], sharex=scatter_ax)
    hist_y_ax = fig.add_subplot(subgrid1[1, 1], sharey=scatter_ax)
    
    scat_alpha = 0.1
    scatter = scatter_ax.scatter(*images.reshape(2, -1), alpha=scat_alpha, s=1)    
    
    hist_x_ax.hist(images[0].flatten(), bins=int(images.shape[-1]))
    hist_y_ax.hist(
        images[1].flatten(),
        bins=int(images.shape[-1]),
        orientation='horizontal'
    )
    
    rug_params = dict(lw=0.4, alpha=scat_alpha, c='k',  s=100, zorder=3)
    rug_x = hist_x_ax.scatter(
        images[0].flatten(),
        0*images[0].flatten(),
        marker='|',
        **rug_params
    )
    rug_y = hist_y_ax.scatter(
        0*images[1].flatten(),
        images[1].flatten(),
        marker='_',
        **rug_params
    )
    
    scatter_ax.set(
        xlabel='$A$',
        ylabel='$B$',
    )
    for side in ('top', 'right'):
        scatter_ax.spines[side].set_visible(False)
    
    hist_x_ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        hist_x_ax.spines[side].set_visible(False)
    
    hist_y_ax.set_xticks([])
    for side in ('top', 'right', 'bottom'):
        hist_y_ax.spines[side].set_visible(False)

    for ax in material_axes+[im_ax0, im_ax1]:
        ax.set_axis_off()
    for cbar in (cbar0, cbar1):
        cbar.outline.set_visible(False)
    
    fig.tight_layout()
    return fig

def plot_KMeans1D(image, k, width=12, **kwargs):
        
    kmeans = KMeans(k, n_init=100, **kwargs).fit(image.reshape(-1, 1))
    labels = kmeans.labels_.reshape(image.shape)
    means = kmeans.cluster_centers_.flatten()
    
    # Get maps of individual clusters in order of ascending mean
    sorter = np.argsort(means)
    clusters = [labels==sorter[i] for i in range(k)]
    
    means = np.sort(means)
    bounds = np.zeros(k+1)
    bounds[0] = image.min()
    bounds[1:-1] = (means[:-1] + means[1:]) / 2
    bounds[-1] = image.max()
    
    fig = plt.figure(figsize=(width, width/2 + width/(2*k)))
    gs = plt.GridSpec(
        figure=fig,
        nrows=2,
        ncols=k+1,
        width_ratios=k*[1/k,] + [1,]
    )
    
    im_ax = fig.add_subplot(gs[0, :k])
    im = im_ax.imshow(image, cmap=plt.cm.Spectral)
    im_ax.set_title(f'image')
    cbar = fig.colorbar(im, ax=im_ax)
    
    cluster_axes = [fig.add_subplot(gs[1, i]) for i in range(k)]
    for i, (ax, mat) in enumerate(zip(cluster_axes, clusters)):
        ax.imshow(mat, interpolation='nearest')
        ax.set_title(f'cluster {i+1}', c=f'C{i+1}')
    
    hist_ax = fig.add_subplot(gs[:, -1])
    hist_ax.hist(image.flatten(), bins=int(np.sqrt(image.size)))
    for i in range(k):
        hist_ax.scatter(
            image[clusters[i]],
            0*image[clusters[i]],
            marker='|',
            lw=1,
            alpha=0.5,
            s=500,
            c=f'C{i+1}',
            zorder=3
        )
        hist_ax.axvspan(bounds[i], bounds[i+1], fc=f'C{i+1}', alpha=0.1)
        hist_ax.axvline(means[i], c=f'C{i+1}', ls='--')
    hist_ax.set(
        title=f'values',
        yticks=[]
    )
    for side in ('top', 'left', 'right'):
        hist_ax.spines[side].set_visible(False)

    for ax in cluster_axes+[im_ax,]:
        ax.set_axis_off()
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig, kmeans
    
def plot_GMM1D(image, k, width=12, **kwargs):
        
    gmm = GaussianMixture(n_components=k, n_init=100, **kwargs)
    gmm.fit(image.reshape(-1, 1))
    labels = gmm.predict(image.reshape(-1, 1)).reshape(image.shape)
    weights = gmm.weights_.flatten()
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    
    # Get maps of individual clusters in order of ascending mean
    sorter = np.argsort(means)
    clusters = [labels==sorter[i] for i in range(k)]
    probs = [
        gmm.predict_proba(image.reshape(-1, 1))[:, i].reshape(image.shape)
        for i in sorter
    ]
    
    weights = weights[sorter]
    stds = stds[sorter]
    means = means[sorter]
    
    fig = plt.figure(figsize=(width, width/2 + width/(2*k)))
    gs = plt.GridSpec(
        figure=fig,
        nrows=2,
        ncols=k+1,
        width_ratios=k*[1/k,] + [1,]
    )
    
    im_ax = fig.add_subplot(gs[0, :k])
    im = im_ax.imshow(image, cmap=plt.cm.Spectral)
    im_ax.set_title(f'image')
    cbar = fig.colorbar(im, ax=im_ax)
    
    cluster_axes = [fig.add_subplot(gs[1, i]) for i in range(k)]
    for i, (ax, mat) in enumerate(zip(cluster_axes, probs)):
        ax.imshow(mat, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title( f'$P$(cluster {i+1})', c=f'C{i+1}')
    
    hist_ax = fig.add_subplot(gs[:, -1])
    hist_domain = np.linspace(image.min(), image.max(), 512)
    hist_ax.hist(image.flatten(), bins=int(np.sqrt(image.size)), density=True)
    for i in range(k):
        hist_ax.plot(
            hist_domain,
            (
                Gaussian(hist_domain, means[i], stds[i])
                * weights[i] / (stds[i]*np.sqrt(2*np.pi))
            ),
            c=f'C{i+1}',
            ls='--'
        )
        hist_ax.scatter(
            image[clusters[i]],
            0*image[clusters[i]],
            marker='|',
            lw=1,
            alpha=0.5,
            s=500,
            c=f'C{i+1}',
            zorder=3
        )
    hist_ax.set(
        title=f'values',
        yticks=[]
    )
    for side in ('top', 'left', 'right'):
        hist_ax.spines[side].set_visible(False)

    for ax in cluster_axes+[im_ax,]:
        ax.set_axis_off()
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig, gmm

def confidence_ellipse(ax, mean, cov, **kwargs):
    sigma_x, sigma_y = np.diag(cov)
    pearson = cov[1, 0]/ (sigma_x*sigma_y)
    
    ellipse = Ellipse(
        xy=(0, 0),
        width=1,
        height=1,
        **kwargs
    )
    ax.add_patch(ellipse)
    width = 2 * (1 + pearson)
    height = 2 * (1 - pearson)
    ellipse.set(width=width, height=height)
    
    transform = Affine2D()
    transform.rotate_deg(45)
    transform.scale(2*sigma_x, 2*sigma_y)
    transform.translate(*mean)
    
    ellipse.set_transform(transform+ax.transData)