import os

import matplotlib.pyplot as plt
import numpy as np
from dohl import damage as dmg

if __name__ == '__main__':

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    # read images (screenshots from orange book)
    rep = os.path.join('data')
    im1 = plt.imread(os.path.join(rep, 'fatigue_test_1_layer.png'))
    im2 = plt.imread(os.path.join(rep, 'fatigue_test_2_layers.png'))
    im3 = plt.imread(os.path.join(rep, 'fatigue_test_3_layers.png'))
    img = [im1, im2, im3]

    # -- compare sbl with data points

    fig, ax = plt.subplots(figsize=(12.8, 4.8), nrows=1, ncols=3)
    plt.suptitle('EPRI 2009 data (figures 3.2-23 to 3.2-25)')
    N = np.logspace(5, 9, 41)
    for i in range(3):
        ax[i].set_xscale('log')
        ax[i].set_xlim(N[[0, -1]])
        ax[i].set_xlabel('Num cycles at first wire break')
        ax[i].set_ylabel('Fictive stress (MPa)')
        ax[i].set_title('Fatigue test of %s-layer ACSR' % (['one', 'two', 'three'][i],))
        ax[i].grid(True)
    ax[0].set_ylim([0, 80])
    ax[1].set_ylim([0, 45])
    ax[2].set_ylim([0, 40])
    for i in range(3):
        ax[i].imshow(img[i], aspect='auto', origin="upper", transform=ax[i].transAxes,
                     extent=[0, 1, 0, 1])
        ax[i].semilogx(N, dmg.safe_border_line(N, False), label='single layer acsr')
        ax[i].semilogx(N, dmg.safe_border_line(N, True), label='multi layer acsr')
        ax[i].legend()

    # -- plot safe border line

    N = np.logspace(0, 16, 1601)
    s = np.logspace(-2, +3, 1501)

    plt.figure()
    plt.loglog(N, dmg.safe_border_line(N, False), label='single layer acsr')
    plt.loglog(N, dmg.safe_border_line(N, True), label='multi layer acsr')
    plt.grid(True)
    plt.legend()
    plt.title('Safe border line')
    plt.xlabel('Number of cycles (N)')
    plt.ylabel('Fictive stress ($\sigma$, MPa)')

    # -- plot inverse safe border line

    plt.figure()
    plt.loglog(s, dmg.inv_safe_border_line(s, False), label='single layer acsr')
    plt.loglog(s, dmg.inv_safe_border_line(s, True), label='multi layer acsr')
    plt.grid(True)
    plt.legend()
    plt.title('Inverse Safe border line')
    plt.xlabel('Fictive stress ($\sigma$, MPa)')
    plt.ylabel('Number of cycles (N)')

    # --

    plt.show()
