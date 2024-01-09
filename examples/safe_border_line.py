import numpy as np
from dohl import damage as dmg
import matplotlib.pyplot as plt

def _aster570_data():
    m = 1.571
    D = 3.105E-02
    rts = 1.853E+05
    EImin = 2.828E+01
    EImax = 2.155E+03
    alpha = 0.140652249164
    young = 6.8E+10
    d = 3.45E-03
    ml = False
    return m, D, rts, EImin, EImax, alpha, young, d, ml



if __name__ == '__main__' :

    im1 = plt.imread('./data/fatigue_test_1_layer.png')
    im2 = plt.imread('./data/fatigue_test_2_layers.png')
    im3 = plt.imread('./data/fatigue_test_3_layers.png')
    img = [im1, im2, im3]

    N = np.logspace(5, 9, 41)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(30, 10.5)
    plt.suptitle('EPRI 2009 data (figures 3.2-23 to 3.2-25)')
    for i in range(3):
        ax[i].set_xscale('log')
        ax[i].set_xlim([1E+05, 1E+09])
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

    plt.savefig('compare.png', bbox_inches='tight')
    # --

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

    plt.savefig('sbl.png')

    plt.figure()
    plt.loglog(s, dmg.inv_safe_border_line(s, False), label='single layer acsr')
    plt.loglog(s, dmg.inv_safe_border_line(s, True), label='multi layer acsr')
    plt.grid(True)
    plt.legend()
    plt.title('Inverse Safe border line')
    plt.xlabel('Fictive stress ($\sigma$, MPa)')
    plt.ylabel('Number of cycles (N)')

    plt.savefig('isbl.png')