"""
Widget to show the different parameters of a covariance matrix.

Based on maths from:
https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

from functions import Gaussian


def covariance_matrix(N_points=512, plotting_range=10, N_hist_points=256):
    """
    Widget to show the different parameters of a covariance matrix.

    Based on maths from:
    https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    """
    # Generate some random points with no covariance
    xy_points = np.random.randn(2, N_points)

    # Plotting domain for the histograms
    xy_hist = np.linspace(-plotting_range, plotting_range, N_hist_points)

    # Plotting
    fig = plt.figure(figsize=(9, 9))

    # Set up axes geometry and appearance
    gs = plt.GridSpec(
        figure=fig,
        nrows=5, ncols=5,
        width_ratios=(3, 3, 0.1, 1, 0.1),
        height_ratios=(0.1, 1, 0.1, 3, 3),
    )

    scatter_ax = fig.add_subplot(gs[-2:, :2])
    scatter_ax.set(
        xlabel='$x$',
        ylabel='$y$',
        xlim=(-plotting_range, plotting_range),
        ylim=(-plotting_range, plotting_range),
        aspect='equal'
    )
    for side in ('top', 'right'):
        scatter_ax.spines[side].set_visible(False)

    hist_x_ax = fig.add_subplot(gs[1, :2], sharex=scatter_ax)
    hist_y_ax = fig.add_subplot(gs[-2:, -2], sharey=scatter_ax)
    hist_x_ax.set_ylim(-0.05, 1.05)
    hist_y_ax.set_xlim(-0.05, 1.05)
    hist_x_ax.set_axis_off()
    hist_y_ax.set_axis_off()

    slider_x_ax = fig.add_subplot(gs[2, :2])
    slider_y_ax = fig.add_subplot(gs[-2:, -3])
    slider_sx_ax = fig.add_subplot(gs[0, 1])
    slider_sy_ax = fig.add_subplot(gs[-2, -1])
    slider_rho_ax = fig.add_subplot(gs[1, -2])

    slider_x = Slider(
        ax=slider_x_ax,
        label='${\mu}_{x}$',
        valmin=-plotting_range,
        valmax=plotting_range,
        valinit=0,
        orientation='horizontal'
    )
    slider_y = Slider(
        ax=slider_y_ax,
        label='${\mu}_{y}$',
        valmin=-plotting_range,
        valmax=plotting_range,
        valinit=0,
        orientation='vertical'
    )
    slider_sx = Slider(
        ax=slider_sx_ax,
        label='${\sigma}_{x}$',
        valmin=1e-12, # Avoids divide by zero errors
        valmax=plotting_range,
        valinit=1,
        orientation='horizontal'
    )
    slider_sy = Slider(
        ax=slider_sy_ax,
        label='${\sigma}_{y}$',
        valmin=1e-12, # Avoids divide by zero errors
        valmax=plotting_range,
        valinit=1,
        orientation='vertical'
    )
    slider_sx.vline.set_visible(False)
    slider_sy.hline.set_visible(False)
    slider_rho = Slider(
        ax=slider_rho_ax,
        label=r'${\rho}$',
        valmin=-1,
        valmax=1,
        valinit=0,
        orientation='horizontal'
    )

    fig.subplots_adjust(
        top=0.98,
        bottom=0.085,
        left=0.085,
        right=0.98,
        hspace=0.2,
        wspace=0.2
    )

    # Initialise plots
    scat_alpha = 0.5
    scatter = scatter_ax.scatter(*xy_points, alpha=scat_alpha)

    rug_params = dict(lw=0.4, alpha=scat_alpha)
    rug_x = hist_x_ax.scatter(*xy_points, marker='|', **rug_params)
    rug_y = hist_y_ax.scatter(*xy_points, marker='_', **rug_params)

    line_x, = hist_x_ax.plot(xy_hist, xy_hist)
    line_y, = hist_y_ax.plot(xy_hist, xy_hist)

    sigma_ls, sigma_c = '--', 'r'
    sigma_x_plus = hist_x_ax.axvline(0, c=sigma_c, ls=sigma_ls)
    sigma_x_minus = hist_x_ax.axvline(0, c=sigma_c, ls=sigma_ls)
    sigma_y_plus = hist_y_ax.axhline(0, c=sigma_c, ls=sigma_ls)
    sigma_y_minus = hist_y_ax.axhline(0, c=sigma_c, ls=sigma_ls)

    ellipse = Ellipse(
        xy=(0, 0),
        width=1,
        height=1,
        fc='none',
        ec=sigma_c,
        ls=sigma_ls,
        zorder=3
    )
    scatter_ax.add_patch(ellipse)

    def update(val):
        # update mean and covariance matrix from sliders
        mean = np.array([slider_x.val, slider_y.val])
        sigma_x, sigma_y = slider_sx.val, slider_sy.val
        pearson = slider_rho.val
        cov = np.array([
            [sigma_x**2, pearson*sigma_x*sigma_y],
            [pearson*sigma_x*sigma_y, sigma_y**2]
        ])
        
        # Update points based on mean and covariance
        M = np.array([
            [(1+pearson)*sigma_x, (pearson-1)*sigma_x],
            [(1+pearson)*sigma_y, (1-pearson)*sigma_y]
        ]) * np.sqrt(2)
        xy = mean + (M@xy_points).T
        scatter.set_offsets(xy)
        rug_x.set_offsets(xy*np.array([1, 0]))
        rug_y.set_offsets(xy*np.array([0, 1]))
        
        # Update markers
        line_x.set_ydata(Gaussian(xy_hist, mean[0], sigma_x))
        line_y.set_xdata(Gaussian(xy_hist, mean[1], sigma_y))
        sigma_x_plus.set_xdata((mean[0]+sigma_x)*np.array([1, 1]))
        sigma_x_minus.set_xdata((mean[0]-sigma_x)*np.array([1, 1]))
        sigma_y_plus.set_ydata((mean[1]+sigma_y)*np.array([1, 1]))
        sigma_y_minus.set_ydata((mean[1]-sigma_y)*np.array([1, 1]))
        
        # And ellipse
        width = 2 * (1 + pearson)
        height = 2 * (1 - pearson)
        ellipse.set(width=width, height=height)
        
        transform = Affine2D()
        transform.rotate_deg(45)
        transform.scale(2*sigma_x, 2*sigma_y)
        transform.translate(*mean)
            
        ellipse.set_transform(transform+scatter_ax.transData)
        
        fig.canvas.draw_idle()

    for slider in (slider_x, slider_y, slider_sx, slider_sy, slider_rho):
        slider.on_changed(update)

    update(None) # Make sure initial values match sliders
    return fig

if __name__=='__main__':
    fig = covariance_matrix()
    plt.show()