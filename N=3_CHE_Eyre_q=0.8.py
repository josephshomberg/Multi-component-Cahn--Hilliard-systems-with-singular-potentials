###############################################################################
# first-order implicit finite-difference method (with backward Euler) solution of 2D ternary CHE with primary data
# backward Euler with Newton-Ralphson and convex-concave splitting
###############################################################################

# to make an animation, run from terminal in the image directory
# ffmpeg -r 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -r 10 CHE-2D-ternary-BK.mp4

# load libraries
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylab as pylab
from numpy import savez_compressed
import csv

# store energy values to a csv file
with open('data/energy[n].csv', mode='w', newline='') as en_file:
    csvwriter = csv.writer(en_file, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

# store energy values to a csv file
with open('data/rel_error[n].csv', mode='w', newline='') as er_file:
    csvwriter = csv.writer(er_file, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

# #############################################################################
# Setting constants
# #############################################################################
tsallis = 0.8
START = 0
END = 1
N = 50000  # number of time steps / iterations:: N*deltat=10000*0.0001=1
pictureseverywhen = 100  # to make images on multiples of X
deltat = 0.0060  # with ep**2 = (0.008)(0.0005) = 0.000004
T = N * deltat  # end time / time interval length
Lx = 1.0  # right end-point of symmetric interval [-L,L]
Ly = 1.0  # right end-point of symmetric interval [-L,L]
M  = 64  # number of space steps: subinterval length deltax=2L/M
hx = 2.0 * Lx / M  # to determine spatial step sizes
hy = 2.0 * Ly / M  # to determine spatial step sizes
h = hx * hy  # area element for 2D integration
#domain_discretize_x = np.linspace(-L, L, M)  # to generate a meshgrid for the domain
#domain_discretize_y = np.linspace(-L, L, M)
#X, Y = np.meshgrid(domain_discretize_x, domain_discretize_y)
IMAX = 200
implicit_tol = 1e-10
ep = 0.022360679774998  # with 0.022360679774998, ep**2 = 0.0005
lockappa = 1.0 * ep ** 2  # diffusivity of chemical potential
locgamma = 1.3 * ep ** 2  # 1.0 # the diffusivity of the order-parameter
loctheta = 1.0  # coefficient on log nonlinear terms
loctheta_c = 3.4  # coefficient on mixed nonlinear terms (exchange_factor)
# when loctheta / loctheta_c < 1, we have--in the binary case--a two phase binary mixture and two local minima
#cutoff = 0.99 # 0.96  # this cutoff only applies to psi in the auxiliary problem
## WARNING: cutoff > 1 - min of initial data (e.g. 1 - 0.18587028306910447 = 0.81412972)
#lowercutoffval = np.log(1.0 - cutoff)
#uppercutoffval = np.log(cutoff)
pos_tol = 0.001  # 1e-4  # energy slope positivity tolerance
#tol = 1e-4  # when computing relative error
eps = 0  # Model 1 = 0; Model 2 perturbation of the mobility matrix

# #############################################################################
# Constant mobility matrix
# #############################################################################
# MobMa = np.array([[2, 1, -3], [1, 1, -2 ], [-3, -2, 5]])  # symmetric, 1D nullspace, |L|=7.60555
#MobMa = (1.0 / 3) * np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])  # BloCopEll96 / symmetric, 1D nullspace, |L|=1 (the largest singular value = max sqrt( eigenvalues(MT.M) ) )
#MobMa = (1.0 / 3) * np.array([[2 + eps, -1, -1], [-1, 2, -1], [-1 + eps, -1, 2]])  # perturbation
MobMa = ( 1.0 / ( 0.5 * ( 3 + eps + np.sqrt(9 - 2 * eps + eps ** 2) ) ) ) * np.array([[1, -1, eps], [-1, 2, -1], [eps, -1, 1]])  # |L|=1 Model 1/Model2 notes

# initialize functions
locmu  = np.zeros((3, N + 1, M, M))
locphi = np.zeros((3, N + 1, M, M))
psi    = np.zeros((3, IMAX, M, M))
ppsi   = np.zeros((3, IMAX, M, M))
psimu  = np.zeros((3, IMAX, M, M))

# #############################################################################
# locphi[:, 0, :, :] is made here
# #############################################################################

# this is where we make 'near-equilibrium' or 'noisy' initial data according to specific RGB proportions
# C1pro = 0.49  # Color 1 proportion
# C2pro = 0.49  # Color 2 proportion
# C3pro = 0.02  # Color 3 proportion

# C1pro = 0.43  # Color 1 proportion
# C2pro = 0.43  # Color 2 proportion
# C3pro = 0.14  # Color 3 proportion

# C1pro = 0.39  # Color 1 proportion
# C2pro = 0.39  # Color 2 proportion
# C3pro = 0.22  # Color 3 proportion

# C1pro = 0.34  # Color 1 proportion
# C2pro = 0.33  # Color 2 proportion
# C3pro = 0.33  # Color 3 proportion

C1pro = 0.25  # Color 1 proportion
C2pro = 0.25  # Color 2 proportion
C3pro = 0.50  # Color 3 proportion

#C1pro = 0.15  # Color 1 proportion
#C2pro = 0.15  # Color 2 proportion
#C3pro = 0.70  # Color 3 proportion

# # to plot the current proportions
# pylab.clf()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# for m in range(0, M):
#     for l in range(0, M):
#         ax.scatter(m, l, color=phi[:, m, l], s=SZ, marker='s')  # s=50,13
# # plt.show()
# plt.savefig('images1CD/proportions_phi00000.png')

## 'near-equilibrium'
## each row and column must sum to 1
#for m in range(0, M):
#    for l in range(0, int(np.ceil(C1pro * M))):
#        locphi[0, 0, m, l] = 0.9  # * np.random.rand()   # RED
#        locphi[1, 0, m, l] = 0.05  # * np.random.rand()  # GREEN
#        locphi[2, 0, m, l] = 1 - locphi[0, 0, m, l] - locphi[1, 0, m, l]  # BLUE
#for m in range(0, M):
#    for l in range(int(np.ceil(C1pro * M)), int(np.ceil(C1pro * M)) + int(np.ceil(C2pro * M))):
#        locphi[0, 0, m, l] = 0.05  # * np.random.rand()  # RED
#        locphi[1, 0, m, l] = 0.9  # * np.random.rand()   # GREEN
#        locphi[2, 0, m, l] = 1 - locphi[0, 0, m, l] - locphi[1, 0, m, l]  # BLUE
#for m in range(0, M):
#    for l in range(int(np.ceil(C1pro * M)) + int(np.ceil(C2pro * M)), int(np.ceil(C1pro * M)) + int(np.ceil(C2pro * M)) + int(np.ceil(C3pro * M)) - 0):
#        locphi[0, 0, m, l] = 0.05  # * np.random.rand()  # RED
#        locphi[1, 0, m, l] = 0.05  # * np.random.rand()  # GREEN
#        locphi[2, 0, m, l] = 1 - locphi[0, 0, m, l] - locphi[1, 0, m, l]  # BLUE

# print('the dimensions of phi0', IC.shape)

## to make 'noisy' initial conditions
## each row and column must sum to 1
#for m in range(0, M):
#    for l in range(0, M):
#        locphi[0, m, l] = np.random.normal(C1pro, 0.02)   # * np.random.rand()   # RED
#        locphi[1, m, l] = np.random.normal(C2pro, 0.02)   # * np.random.rand()  # GREEN
#        locphi[2, m, l] = 1 - locphi[0, m, l] - locphi[1, m, l]  # BLUE

## to make Eyer's 'noisy' initial conditions
## each row and column must sum to 1
#for m in range(0, M):
#    for l in range(0, M):
#        locphi[0, m, l] = 0.25 + np.random.uniform(-0.01, 0.01)   # RED
#        locphi[1, m, l] = 0.25 + np.random.uniform(-0.01, 0.01)   # GREEN
#        locphi[2, m, l] = 1 - locphi[0, m, l] - locphi[1, m, l]  # BLUE

## to save initial data to csv (comment out if wanting to save after the shuffle)
#np.savetxt('data/initial_data0red.csv', locphi[0, 0, :, :], delimiter=',')
#np.savetxt('data/initial_data1green.csv', locphi[1, 0, :, :], delimiter=',')
#np.savetxt('data/initial_data2blue.csv', locphi[2, 0, :, :], delimiter=',')

# to load csv initial data
with open('data/initial_data0red.csv', 'r') as red:
    reader = csv.reader(red, delimiter=',')
    data = np.array(list(reader)).astype(float)
locphi[0, 0, :, :] = data[:, :]  # red initial data
with open('data/initial_data1green.csv', 'r') as green:
    reader = csv.reader(green, delimiter=',')
    data = np.array(list(reader)).astype(float)
locphi[1, 0, :, :] = data[:, :]  # green initial data
with open('data/initial_data2blue.csv', 'r') as blue:
    reader = csv.reader(blue, delimiter=',')
    data = np.array(list(reader)).astype(float)
locphi[2, 0, :, :] = data[:, :]  # blue initial data

## to load npz initial data
#from numpy import load
#def load_IC(filename):
#    # load compressed arrays
#    data = load(filename, mmap_mode='r')
#    # unpack arrays
#    IC = data['arr_0']  #
##    IC = np.moveaxis(IC, 2, 0)
#    # check for 'nan' values in data
#    assert not np.any(np.isnan(IC))
#    return IC
#
#initial = load_IC('40-40-20-Eyre_train=0.npz')
#print('Loaded IC image', initial[0].shape)
#phi[:, 0, :, :] = initial[0][:, :, :]
#print('Made phi0 image', phi[:, 0, :, :].shape)

#print('phi[:, 0, :, :].shape', phi[:, 0, :, :].shape)
#print('np.moveaxis(phi[:, 0, :, :], 0, -1).shape', np.moveaxis(phi[:, 0, :, :], 0, -1).shape)

## to plot the current proportions
#pylab.clf()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_aspect('equal')
#ax.imshow(np.moveaxis(phi[:, 0, :, :], 0, -1))
#plt.axis('off')  # turn off the axis labels
## plt.show()
#plt.savefig('images/proportions_phi00000.png', bbox_inches='tight', pad_inches=0)

# #############################################################################
# Set various functions
# #############################################################################

def TED(q, x):
    assert q > 0
    if q == 1.0:
        return x * np.log(x)
    if q != 1.0:
        return x * (x ** (1 - q) - 1.0) / (1 - q)


def DTED(q, x):
    assert q > 0
    if q == 1.0:
        return np.log(x) + 1.0
    if q != 1.0:
        return ((2 - q) * x ** (1 - q) - 1) / (1 - q)


# used with mu[0] and mu[n] ############################################

def Lapphi(n, m, l): return (locphi[:, n, m + 1, l] + locphi[:, n, m - 1, l] - 2 * locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] + locphi[:, n, m, l - 1] - 2 * locphi[:, n, m, l]) / (hy**2)


def DownLapphi(n, m, l): return (locphi[:, n, m + 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] + locphi[:, n, m, l - 1] - 2 * locphi[:, n, m, l]) / (hy**2)  # when m=0 used as RightLapphi(:, n-1, 0, l)


def UpLapphi(n, m, l): return (locphi[:, n, m - 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] + locphi[:, n, m, l - 1] - 2 * locphi[:, n, m, l]) / (hy**2)  # when m=M-1 used as LeftLapphi(:, n-1, M-1, l)


def LeftLapphi(n, m, l): return (locphi[:, n, m + 1, l] + locphi[:, n, m - 1, l] - 2 * locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l - 1] - locphi[:, n, m, l]) / (hy**2)  # when l=0 used as UpLapphi(:, n-1, m, 0)


def RightLapphi(n, m, l): return (locphi[:, n, m + 1, l] + locphi[:, n, m - 1, l] - 2 * locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] - locphi[:, n, m, l]) / (hy**2)  # when l=M-1 used as DownLapphi(:, n-1, m, M-1)


def NWLapphi(n, m, l): return (locphi[:, n, m + 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] - locphi[:, n, m, l]) / (hy**2)  # when m=0 used as RightLapphi(:, n-1, 0, l)


def SWLapphi(n, m, l): return (locphi[:, n, m - 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l + 1] - locphi[:, n, m, l]) / (hy**2)  # when m=M-1 used as LeftLapphi(:, n-1, M-1, l)


def SELapphi(n, m, l): return (locphi[:, n, m - 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l - 1] - locphi[:, n, m, l]) / (hy**2)  # when l=0 used as UpLapphi(:, n-1, m, 0)


def NELapphi(n, m, l): return (locphi[:, n, m + 1, l] - locphi[:, n, m, l]) / (hx**2) + (locphi[:, n, m, l - 1] - locphi[:, n, m, l]) / (hy**2)  # when l=M-1 used as DownLapphi(:, n-1, m, M-1)


# used with Convex ############################################

def CLappsi(c, m, l): return (psi[:, c, m + 1, l] + psi[:, c, m - 1, l] - 2 * psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] + psi[:, c, m, l - 1] - 2 * psi[:, c, m, l]) / (hy**2)


def DownCLappsi(c, m, l): return (psi[:, c, m + 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] + psi[:, c, m, l - 1] - 2 * psi[:, c, m, l]) / (hy**2)  # when m=0 used as RightLapphi(:, n-1, 0, l)


def UpCLappsi(c, m, l): return (psi[:, c, m - 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] + psi[:, c, m, l - 1] - 2 * psi[:, c, m, l]) / (hy**2)  # when m=M-1 used as LeftLapphi(:, n-1, M-1, l)


def LeftCLappsi(c, m, l): return (psi[:, c, m + 1, l] + psi[:, c, m - 1, l] - 2 * psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l - 1] - psi[:, c, m, l]) / (hy**2)  # when l=0 used as UpLapphi(:, n-1, m, 0)


def RightCLappsi(c, m, l): return (psi[:, c, m + 1, l] + psi[:, c, m - 1, l] - 2 * psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] - psi[:, c, m, l]) / (hy**2)  # when l=M-1 used as DownLapphi(:, n-1, m, M-1)


def NWCLappsi(c, m, l): return (psi[:, c, m + 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] - psi[:, c, m, l]) / (hy**2)


def SWCLappsi(c, m, l): return (psi[:, c, m - 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l + 1] - psi[:, c, m, l]) / (hy**2)


def SECLappsi(c, m, l): return (psi[:, c, m - 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l - 1] - psi[:, c, m, l]) / (hy**2)


def NECLappsi(c, m, l): return (psi[:, c, m + 1, l] - psi[:, c, m, l]) / (hx**2) + (psi[:, c, m, l - 1] - psi[:, c, m, l]) / (hy**2)


# used with LappsimuMa #########################################

def Lappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] + psimu[i, c, m - 1, l] - 2 * psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] + psimu[i, c, m, l - 1] - 2 * psimu[i, c, m, l]) / (hy**2)


def DownLappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] + psimu[i, c, m, l - 1] - 2 * psimu[i, c, m, l]) / (hy**2)


def UpLappsimu(i, c, m, l): return (psimu[i, c, m - 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] + psimu[i, c, m, l - 1] - 2 * psimu[i, c, m, l]) / (hy**2)


def LeftLappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] + psimu[i, c, m - 1, l] - 2 * psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l - 1] - psimu[i, c, m, l]) / (hy**2)


def RightLappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] + psimu[i, c, m - 1, l] - 2 * psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] - psimu[i, c, m, l]) / (hy**2)


def NWLappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] - psimu[i, c, m, l]) / (hy**2)


def SWLappsimu(i, c, m, l): return (psimu[i, c, m - 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l + 1] - psimu[i, c, m, l]) / (hy**2)


def SELappsimu(i, c, m, l): return (psimu[i, c, m - 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l - 1] - psimu[i, c, m, l]) / (hy**2)


def NELappsimu(i, c, m, l): return (psimu[i, c, m + 1, l] - psimu[i, c, m, l]) / (hx**2) + (psimu[i, c, m, l - 1] - psimu[i, c, m, l]) / (hy**2)


# used with psi[i] #########################################

def LappsimuMa(c, m, l): return np.array([Lappsimu(0, c, m, l), Lappsimu(1, c, m, l), Lappsimu(2, c, m, l)])  # 1x3 matrix


def DownLappsimuMa(c, m, l): return np.array([DownLappsimu(0, c, m, l), DownLappsimu(1, c, m, l), DownLappsimu(2, c, m, l)])  # 1x3 matrix


def UpLappsimuMa(c, m, l): return np.array([UpLappsimu(0, c, m, l), UpLappsimu(1, c, m, l), UpLappsimu(2, c, m, l)])  # 1x3 matrix


def LeftLappsimuMa(c, m, l): return np.array([LeftLappsimu(0, c, m, l), LeftLappsimu(1, c, m, l), LeftLappsimu(2, c, m, l)])  # 1x3 matrix


def RightLappsimuMa(c, m, l): return np.array([RightLappsimu(0, c, m, l), RightLappsimu(1, c, m, l), RightLappsimu(2, c, m, l)])  # 1x3 matrix


def NWLappsimuMa(c, m, l): return np.array([NWLappsimu(0, c, m, l), NWLappsimu(1, c, m, l), NWLappsimu(2, c, m, l)])  # 1x3 matrix


def SWLappsimuMa(c, m, l): return np.array([SWLappsimu(0, c, m, l), SWLappsimu(1, c, m, l), SWLappsimu(2, c, m, l)])  # 1x3 matrix


def SELappsimuMa(c, m, l): return np.array([SELappsimu(0, c, m, l), SELappsimu(1, c, m, l), SELappsimu(2, c, m, l)])  # 1x3 matrix


def NELappsimuMa(c, m, l): return np.array([NELappsimu(0, c, m, l), NELappsimu(1, c, m, l), NELappsimu(2, c, m, l)])  # 1x3 matrix


# Convex ################################################
# Implicit part: Tsallis entropy derivative only.

def Convex(c, m, l):
    return loctheta * DTED(tsallis, psi[:, c, m, l])


# Concave / explicit part ################################
# The Laplacian term has been moved out of Convex.
# Both the -gamma*Delta(phi) term and the interaction term are evaluated
# at the OLD TIME LEVEL n-1, not at the previous fixed-point iterate.

def Concave_bulk(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * Lapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_down(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * DownLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_up(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * UpLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_left(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * LeftLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_right(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * RightLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_NW(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * NWLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_SW(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * SWLapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_SE(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * SELapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


def Concave_NE(n, m, l):
    old = locphi[:, n - 1, m, l]
    return (
        -locgamma * NELapphi(n - 1, m, l)
        + loctheta_c * (old[[2, 0, 1]] + old[[1, 2, 0]])
    )


# Energy ###############################################################

def Energy_1(n):
    return 0.5 * locgamma * h * (
    
    np.sum( ( (locphi[:, n, 1 + 1:M-1 + 1, 1:M-1] - locphi[:, n, 1 - 1:M-1 - 1, 1:M-1]) / (2*hx) ) ** 2 )
    
    + np.sum( ( (locphi[:, n, 1:M-1, 1 + 1:M-1 + 1] - locphi[:, n, 1:M-1, 1 - 1:M-1 - 1]) / (2*hy) ) ** 2 )
    
    + np.sum( ( (locphi[:, n, :, M-1] - locphi[:, n, :, M-1 - 1]) / hy ) ** 2 )
     
    + np.sum( ( (locphi[:, n, :, 0 + 1] - locphi[:, n, :, 0]) / hy ) ** 2 )
      
    + np.sum( ( (locphi[:, n, M-1, :] - locphi[:, n, M-1 - 1, :]) / hx ) ** 2 )
    
    + np.sum( ( (locphi[:, n, 0 + 1, :] - locphi[:, n, 0, :]) / hx ) ** 2 )
    
    + np.sum( ( (locphi[:, n, 0+1, 0] - locphi[:, n, 0, 0]) / hx ) ** 2 + ((locphi[:, n, 0, 0+1] - locphi[:, n, 0, 0]) / hy ) ** 2 )

    + np.sum( ( (locphi[:, n, M-1-1, 0] - locphi[:, n, M-1, 0]) / (-hx) ) ** 2 + ((locphi[:, n, M-1, 0+1] - locphi[:, n, M-1, 0]) / hy ) ** 2 )

    + np.sum( ( (locphi[:, n, M-1-1, M-1] - locphi[:, n, M-1, M-1]) / (-hx) ) ** 2 + ((locphi[:, n, M-1, M-1-1] - locphi[:, n, M-1, M-1]) / (-hy) ) ** 2 )

    + np.sum( ( (locphi[:, n, 0+1, M-1] - locphi[:, n, 0, M-1]) / hx ) ** 2 + ((locphi[:, n, 0, M-1-1] - locphi[:, n, 0, M-1]) / (-hy) ) ** 2 )

     )


def Energy_2(n): return loctheta * h * np.sum( TED(tsallis, locphi[0, n]) + TED(tsallis, locphi[1, n]) + TED(tsallis, locphi[2, n]) )  # logs ~ convex i


# no 0.5 below because loctheta_c=3.4 not 6.8...
def Energy_3(n): return loctheta_c * h * np.sum( locphi[0, n] * locphi[1, n] + locphi[1, n] * locphi[2, n] + locphi[2, n] * locphi[0, n] )  # interaction terms ~ concave i-1


def ENERGY(n): return Energy_1(n) + Energy_2(n) + Energy_3(n)


def relative_error_l2(n, phi):
    a = h * np.sum( (phi[:, n, :, :] - phi[:, n - 1, :, :]) ** 2 )
    b = h * np.sum( phi[:, n, :, :] ** 2 )
    relative_error = a / b
    return relative_error


################################################################################
# BEGIN MAIN LOOP HERE
################################################################################
count    = START
MAXcount = END
src_list = np.zeros((MAXcount, 3, M, M))  # src_list[iter] = nonphi[:, N, :, :]
tar_list = np.zeros((MAXcount, 3, M, M))  # tar_list[iter] = nonphi[:, 0, :, :]

while count < MAXcount:

    # initializing order-parameter component means
    mean0 = np.zeros(N + 1)
    mean1 = np.zeros(N + 1)
    mean2 = np.zeros(N + 1)
    rconer = np.zeros(N + 1)
    gconer = np.zeros(N + 1)
    bconer = np.zeros(N + 1)

    # check initial data has correct component means and that each pixel sums to 1
    mean0[0] = np.mean(locphi[0, 0])
    mean1[0] = np.mean(locphi[1, 0])
    mean2[0] = np.mean(locphi[2, 0])
    rconer[0] = abs(mean0[0]-mean0[0])
    gconer[0] = abs(mean1[0]-mean1[0])
    bconer[0] = abs(mean2[0]-mean2[0])

    # initializing energies
    e1 = np.zeros(N + 1)
    e2 = np.zeros(N + 1)
    e3 = np.zeros(N + 1)
    en = np.zeros(N + 1)
    locmax = np.zeros(N + 1)
    locmin = np.zeros(N + 1)
    mumax = np.zeros(N + 1)
    mumin = np.zeros(N + 1)
    rel_e  = np.zeros(N + 1)

#    # Shuffling initial data and checking the component means and pixel sums
#    new_indices_1 = np.random.permutation(M * M)
#    holder0 = locphi[0].ravel()
#    holder1 = locphi[1].ravel()
#    holder2 = locphi[2].ravel()
#    holder0 = holder0[new_indices_1]
#    holder1 = holder1[new_indices_1]
#    holder2 = holder2[new_indices_1]
#    locphi[0, :, :] = holder0.reshape(M, M)
#    locphi[1, :, :] = holder1.reshape(M, M)
#    locphi[2, :, :] = holder2.reshape(M, M)
#    mean0[0] = np.mean(locphi[0, 0])
#    mean1[0] = np.mean(locphi[1, 0])
#    mean2[0] = np.mean(locphi[2, 0])
#
#    # to save initial data to csv
#    np.savetxt('data/initial_data0red.csv', locphi[0, 0, :, :], delimiter=',')
#    np.savetxt('data/initial_data1green.csv', locphi[1, 0, :, :], delimiter=',')
#    np.savetxt('data/initial_data2blue.csv', locphi[2, 0, :, :], delimiter=',')

    # initial condition plot RED
    pylab.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.imshow(np.moveaxis(locphi[0, 0, :, :], 0, -1), cmap='Reds')
    plt.axis('off')  # turn off the axis labels
    # plt.show()
    plt.savefig('images/0red/phi' + str(count).zfill(4) + '00000.png', bbox_inches='tight', pad_inches=0)

    # initial condition plot GREEN
    pylab.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.imshow(np.moveaxis(locphi[1, 0, :, :], 0, -1), cmap='Greens')
    plt.axis('off')  # turn off the axis labels
    # plt.show()
    plt.savefig('images/1green/phi' + str(count).zfill(4) + '00000.png', bbox_inches='tight', pad_inches=0)

    # initial condition plot BLUE
    pylab.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.imshow(np.moveaxis(locphi[2, 0, :, :], 0, -1), cmap='Blues')
    plt.axis('off')  # turn off the axis labels
    # plt.show()
    plt.savefig('images/2blue/phi' + str(count).zfill(4) + '00000.png', bbox_inches='tight', pad_inches=0)

    # initial condition plot
    pylab.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.imshow(np.moveaxis(locphi[:, 0, :, :], 0, -1))
    plt.axis('off')  # turn off the axis labels
    # plt.show()
    plt.savefig('images/phi' + str(count).zfill(4) + '00000.png', bbox_inches='tight', pad_inches=0)

    tar_list[count] = locphi[:, 0, :, :]

    ############################################################################
    # locmu[:, 0, :, :] is made here
    ############################################################################

    # initialization of chemical potential mu[0] (no splitting bc n-1 = -1)

    # bulk mu[0]
    for m in range(1, M - 1):
        for l in range(1, M - 1):  # on [1,62]^2
            
            locmu[:, 0, m, l] = (
                -locgamma * Lapphi(0, m, l)
                + loctheta * DTED(tsallis, locphi[:, 0, m, l])
                + loctheta_c * (locphi[:, 0, m, l][[2, 0, 1]] + locphi[:, 0, m, l][[1, 2, 0]])
            )

    # edges mu[0]
    for l in range(1, M - 1):
        locmu[:, 0, 0, l] = (
            -locgamma * DownLapphi(0, 0, l)
            + loctheta * DTED(tsallis, locphi[:, 0, 0, l])
            + loctheta_c * (locphi[:, 0, 0, l][[2, 0, 1]] + locphi[:, 0, 0, l][[1, 2, 0]])
        )
        locmu[:, 0, M - 1, l] = (
            -locgamma * UpLapphi(0, M - 1, l)
            + loctheta * DTED(tsallis, locphi[:, 0, M - 1, l])
            + loctheta_c * (locphi[:, 0, M - 1, l][[2, 0, 1]] + locphi[:, 0, M - 1, l][[1, 2, 0]])
        )

    for m in range(1, M - 1):
        locmu[:, 0, m, 0] = (
            -locgamma * RightLapphi(0, m, 0)
            + loctheta * DTED(tsallis, locphi[:, 0, m, 0])
            + loctheta_c * (locphi[:, 0, m, 0][[2, 0, 1]] + locphi[:, 0, m, 0][[1, 2, 0]])
        )
        locmu[:, 0, m, M - 1] = (
            -locgamma * LeftLapphi(0, m, M - 1)
            + loctheta * DTED(tsallis, locphi[:, 0, m, M - 1])
            + loctheta_c * (locphi[:, 0, m, M - 1][[2, 0, 1]] + locphi[:, 0, m, M - 1][[1, 2, 0]])
        )

    # corners mu[0]
    locmu[:, 0, 0, 0] = (
        -locgamma * NWLapphi(0, 0, 0)
        + loctheta * DTED(tsallis, locphi[:, 0, 0, 0])
        + loctheta_c * (locphi[:, 0, 0, 0][[2, 0, 1]] + locphi[:, 0, 0, 0][[1, 2, 0]])
    )
    locmu[:, 0, M - 1, 0] = (
        -locgamma * SWLapphi(0, M - 1, 0)
        + loctheta * DTED(tsallis, locphi[:, 0, M - 1, 0])
        + loctheta_c * (locphi[:, 0, M - 1, 0][[2, 0, 1]] + locphi[:, 0, M - 1, 0][[1, 2, 0]])
    )
    locmu[:, 0, M - 1, M - 1] = (
        -locgamma * SELapphi(0, M - 1, M - 1)
        + loctheta * DTED(tsallis, locphi[:, 0, M - 1, M - 1])
        + loctheta_c * (locphi[:, 0, M - 1, M - 1][[2, 0, 1]] + locphi[:, 0, M - 1, M - 1][[1, 2, 0]])
    )
    locmu[:, 0, 0, M - 1] = (
        -locgamma * NELapphi(0, 0, M - 1)
        + loctheta * DTED(tsallis, locphi[:, 0, 0, M - 1])
        + loctheta_c * (locphi[:, 0, 0, M - 1][[2, 0, 1]] + locphi[:, 0, 0, M - 1][[1, 2, 0]])
    )
    
    # initial energy and max/min
    e1[0] = Energy_1(0)
    e2[0] = Energy_2(0)
    e3[0] = Energy_3(0)
    en[0] = ENERGY(0)
    locmax[0] = np.max(locphi[:, 0, :, :])
    locmin[0] = np.min(locphi[:, 0, :, :])
    mumax[0] = np.max(locmu[:, 0, :, :])
    mumin[0] = np.min(locmu[:, 0, :, :])

    print('====================================================================')
    print('All pointwise sums = 1.0?', np.allclose(locphi[0, 0] + locphi[1, 0] + locphi[2, 0], 1.0))
    print('Average value of each INITIAL component:', mean0[0], mean1[0], mean2[0], 'the sum of concentrations:', mean0[0] + mean1[0] + mean2[0])
    print('Concentration errors: ', rconer[0], gconer[0], bconer[0], rconer[0] + gconer[0] + bconer[0])
    print('Minimum / maximum value in phi[:,0,:,:]:', locmin[0], locmax[0])
    print('Minimum / maximum value in mu[:,0,:,:]:', mumin[0], mumax[0])
    print('Initial energies', e1[0], e2[0], e3[0], en[0])

    # store energy values to a csv file
    with open('data/energy[n].csv', mode='a', newline='') as en_file:
        csvwriter = csv.writer(en_file, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([e1[0], 0.0, e2[0], 0.0, e3[0], 0.0, e1[0] + e2[0] + e3[0], 0.0])

    # ########################################################################
    # locphi[:, n, :, :] and locmu[:, n, :, :]
    # ########################################################################

    # to make phi[n] and mu[n]
    for n in range(1, N + 1):  # Solution of system and NBC (need locphi[n] and locmu[n])

        # to initialize auxiliary variables
        psi[:, 0, :, :]   = locphi[:, n - 1, :, :]   # initialize psi[n] at each pair (m,l)
        ppsi[:, 0, :, :]  = locphi[:, n - 1, :, :]   # retained for compatibility; concave state uses locphi[n-1]
        psimu[:, 0, :, :] = locmu[:, n - 1, :, :]    # initialize psimu[n] at each pair (m,l)

        # fixed-point / implicit subroutine
        #
        # psi^(i) = phi^(n-1) + dt*kappa*M*Delta(mu^(i-1))
        #
        # mu^(i)  = theta*D(T_q)(psi^(i))
        #           - gamma*Delta(phi^(n-1))
        #           + interaction(phi^(n-1))
        #
        # Thus only the Tsallis term is treated implicitly in the nonlinear
        # iteration.  The Laplacian and interaction terms are lagged at n-1.

        converged = False
        implicit_error = np.inf

        for i in range(1, IMAX):

            # bulk
            for m in range(1, M - 1):
                for l in range(1, M - 1):
                    psi[:, i, m, l] = (
                        locphi[:, n - 1, m, l]
                        + deltat * lockappa
                        * (MobMa @ LappsimuMa(i - 1, m, l).T)
                    )

                    psimu[:, i, m, l] = (
                        Convex(i, m, l)
                        + Concave_bulk(n, m, l)
                    )

            # edges
            for l in range(1, M - 1):
                psi[:, i, 0, l] = (
                    locphi[:, n - 1, 0, l]
                    + deltat * lockappa
                    * (MobMa @ DownLappsimuMa(i - 1, 0, l).T)
                )
                psi[:, i, M - 1, l] = (
                    locphi[:, n - 1, M - 1, l]
                    + deltat * lockappa
                    * (MobMa @ UpLappsimuMa(i - 1, M - 1, l).T)
                )

                psimu[:, i, 0, l] = (
                    Convex(i, 0, l)
                    + Concave_down(n, 0, l)
                )
                psimu[:, i, M - 1, l] = (
                    Convex(i, M - 1, l)
                    + Concave_up(n, M - 1, l)
                )

            for m in range(1, M - 1):
                psi[:, i, m, 0] = (
                    locphi[:, n - 1, m, 0]
                    + deltat * lockappa
                    * (MobMa @ RightLappsimuMa(i - 1, m, 0).T)
                )
                psi[:, i, m, M - 1] = (
                    locphi[:, n - 1, m, M - 1]
                    + deltat * lockappa
                    * (MobMa @ LeftLappsimuMa(i - 1, m, M - 1).T)
                )

                psimu[:, i, m, 0] = (
                    Convex(i, m, 0)
                    + Concave_right(n, m, 0)
                )
                psimu[:, i, m, M - 1] = (
                    Convex(i, m, M - 1)
                    + Concave_left(n, m, M - 1)
                )

            # corners
            psi[:, i, 0, 0] = (
                locphi[:, n - 1, 0, 0]
                + deltat * lockappa
                * (MobMa @ NWLappsimuMa(i - 1, 0, 0).T)
            )
            psi[:, i, M - 1, 0] = (
                locphi[:, n - 1, M - 1, 0]
                + deltat * lockappa
                * (MobMa @ SWLappsimuMa(i - 1, M - 1, 0).T)
            )
            psi[:, i, M - 1, M - 1] = (
                locphi[:, n - 1, M - 1, M - 1]
                + deltat * lockappa
                * (MobMa @ SELappsimuMa(i - 1, M - 1, M - 1).T)
            )
            psi[:, i, 0, M - 1] = (
                locphi[:, n - 1, 0, M - 1]
                + deltat * lockappa
                * (MobMa @ NELappsimuMa(i - 1, 0, M - 1).T)
            )

            psimu[:, i, 0, 0] = (
                Convex(i, 0, 0)
                + Concave_NW(n, 0, 0)
            )
            psimu[:, i, M - 1, 0] = (
                Convex(i, M - 1, 0)
                + Concave_SW(n, M - 1, 0)
            )
            psimu[:, i, M - 1, M - 1] = (
                Convex(i, M - 1, M - 1)
                + Concave_SE(n, M - 1, M - 1)
            )
            psimu[:, i, 0, M - 1] = (
                Convex(i, 0, M - 1)
                + Concave_NE(n, 0, M - 1)
            )

            implicit_error = np.max(
                np.abs(psi[:, i, :, :] - psi[:, i - 1, :, :])
            )

            if implicit_error < implicit_tol:
                converged = True
                break

        if not converged:
            print('================================================================')
            print(
                f'FIXED-POINT FAILURE at n={n}: '
                f'IMAX={IMAX}, last implicit_error={implicit_error:.6e}'
            )
            print('Time step rejected. Stopping simulation.')
            break

        i_last = i

        # Accept the converged fixed-point iterate directly.
        # Do NOT perform one additional phi update after convergence.
        locphi[:, n, :, :] = psi[:, i_last, :, :]
        locmu[:, n, :, :] = psimu[:, i_last, :, :]

        if i_last > 40:
            print(
                f'WARNING: n={n} required {i_last} fixed-point iterations; '
                f'implicit_error={implicit_error:.6e}'
            )

#        # update 'blue' to keep on Gibbs (check concentration errors)
#        locphi[2, n, :, :] = 1 - locphi[0, n, :, :] - locphi[1, n, :, :]

        # corners mu[n]
        locmu[:, n, 0, 0] = (
            -locgamma * NWLapphi(n, 0, 0)
            + loctheta * DTED(tsallis, locphi[:, n, 0, 0])
            + loctheta_c * (locphi[:, n, 0, 0][[2, 0, 1]] + locphi[:, n, 0, 0][[1, 2, 0]])
        )
        locmu[:, n, M - 1, 0] = (
            -locgamma * SWLapphi(n, M - 1, 0)
            + loctheta * DTED(tsallis, locphi[:, n, M - 1, 0])
            + loctheta_c * (locphi[:, n, M - 1, 0][[2, 0, 1]] + locphi[:, n, M - 1, 0][[1, 2, 0]])
        )
        locmu[:, n, M - 1, M - 1] = (
            -locgamma * SELapphi(n, M - 1, M - 1)
            + loctheta * DTED(tsallis, locphi[:, n, M - 1, M - 1])
            + loctheta_c * (locphi[:, n, M - 1, M - 1][[2, 0, 1]] + locphi[:, n, M - 1, M - 1][[1, 2, 0]])
        )
        locmu[:, n, 0, M - 1] = (
            -locgamma * NELapphi(n, 0, M - 1)
            + loctheta * DTED(tsallis, locphi[:, n, 0, M - 1])
            + loctheta_c * (locphi[:, n, 0, M - 1][[2, 0, 1]] + locphi[:, n, 0, M - 1][[1, 2, 0]])
        )
        
        # to check the component means and pixel sums
        mean0[n] = np.mean(locphi[0, n])
        mean1[n] = np.mean(locphi[1, n])
        mean2[n] = np.mean(locphi[2, n])
        # concentration errors
        rconer[n] = abs(mean0[n] - mean0[0])
        gconer[n] = abs(mean1[n] - mean1[0])
        bconer[n] = abs(mean2[n] - mean2[0])
        locmax[n] = np.max(locphi[:, n, :, :])
        locmin[n] = np.min(locphi[:, n, :, :])
        # summarize energy, min/max
        e1[n] = Energy_1(n)
        e2[n] = Energy_2(n)
        e3[n] = Energy_3(n)
        en[n] = ENERGY(n)
        locmax[n] = np.max(locphi[:, n, :, :])
        locmin[n] = np.min(locphi[:, n, :, :])
        mumax[n] = np.max(locmu[:, n, :, :])
        mumin[n] = np.min(locmu[:, n, :, :])
        # compute the relative error in L2-norm
        rel_e[n] = relative_error_l2(n, locphi)

        print('================================================================')
        print('Finished: count =', count + 1, ' of ', MAXcount, ' and iteration n =', n, ' of (max)', N)
        print('All pointwise sums = 1.0?', np.allclose(locphi[0, n] + locphi[1, n] + locphi[2, n], 1.0))
        print('Average value of each component:', mean0[n], mean1[n], mean2[n], 'the sum of concentrations:', mean0[n] + mean1[n] + mean2[n])
        print('Concentration errors:', rconer[n], gconer[n], bconer[n], rconer[n] + gconer[n] + bconer[n])
        print('Minimum / maximum value in phi[:,n,:,:]:', locmin[n], locmax[n])
        print('Minimum / maximum value in mu[:,n,:,:]:', mumin[n], mumax[n])
        print('Current energies:', e1[n], e2[n], e3[n], en[n])
        print('Energy slope:', en[n] - en[n-1])
        print('Current relative error:', rel_e[n])

#        if rel_e[n] <= tol:
#            print('The relative error is within tolerance.')
#        else:
#            print('The relative error is *NOT* within tolerance.')

        # store energy values to a csv file
        with open('data/energy[n].csv', mode='a', newline='') as data_file:
            csvwriter = csv.writer(data_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([e1[n], e1[n] - e1[n - 1], e2[n], e2[n] - e2[n - 1], e3[n], e3[n] - e3[n - 1], en[n], en[n] - en[n - 1]])

        # store energy values to a csv file
        with open('data/rel_error[n].csv', mode='a', newline='') as er_file:
            csvwriter = csv.writer(er_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([rel_e[n]])

        if n % pictureseverywhen == 0:
            # plots some of the sequence RED
            pylab.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(np.moveaxis(locphi[0, n, :, :], 0, -1), cmap='Reds')
            plt.axis('off')  # turn off the axis labels
            # plt.show()
            plt.savefig('images/0red/phi' + str(count).zfill(4) + str(n).zfill(5) + '.png', bbox_inches='tight', pad_inches=0)

            # plots some of the sequence GREEN
            pylab.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(np.moveaxis(locphi[1, n, :, :], 0, -1), cmap='Greens')
            plt.axis('off')  # turn off the axis labels
            # plt.show()
            plt.savefig('images/1green/phi' + str(count).zfill(4) + str(n).zfill(5) + '.png', bbox_inches='tight', pad_inches=0)

            # plots some of the sequence BLUE
            pylab.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(np.moveaxis(locphi[2, n, :, :], 0, -1), cmap='Blues')
            plt.axis('off')  # turn off the axis labels
            # plt.show()
            plt.savefig('images/2blue/phi' + str(count).zfill(4) + str(n).zfill(5) + '.png', bbox_inches='tight', pad_inches=0)

            # plots some of the sequence
            pylab.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(np.moveaxis(locphi[:, n, :, :], 0, -1))
            plt.axis('off')  # turn off the axis labels
            # plt.show()
            plt.savefig('images/phi' + str(count).zfill(4) + str(n).zfill(5) + '.png', bbox_inches='tight', pad_inches=0)

            # concentration error plots
            pylab.clf()
            x = pylab.arange(0, n + 1, 1)
            plt.plot(x, rconer[x], '--', color='r')
            plt.plot(x, gconer[x], '-.', color='g')
            plt.plot(x, bconer[x], ':', color='b')
            plt.plot(x, rconer[x] + gconer[x] + bconer[x], '-', color='black')
            pylab.savefig('data/con_error' + str(count).zfill(4) + '.png')

            # e1 energy plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            plt.plot(x, e1[x], '--', color='r')
            # plt.ylim(energy[x].min()-0.1,energy[x].max()+0.1)
            pylab.savefig('data/energy-e1' + str(count).zfill(4) + '.png')
            
            # e2 energy plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            plt.plot(x, e2[x], '-.', color='g')
            # plt.ylim(energy[x].min()-0.1,energy[x].max()+0.1)
            pylab.savefig('data/energy-e2' + str(count).zfill(4) + '.png')
            
            # e3 energy plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            plt.plot(x, e3[x], ':', color='b')
            # plt.ylim(energy[x].min()-0.1,energy[x].max()+0.1)
            pylab.savefig('data/energy-e3' + str(count).zfill(4) + '.png')
            
            # total energy plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            plt.plot(x, en[x], '-', color='black')
            # plt.ylim(energy[x].min()-0.1,energy[x].max()+0.1)
            pylab.savefig('data/energy-total' + str(count).zfill(4) + '.png')
            
            # energy difference plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            fig, ax = plt.subplots()
            ax.plot(en[x] - en[x - 1])
    #        plt.xlim([0, N])
    #        plt.ylim([-1.5, 0])
    #        legend = ax.legend(loc='lower right', shadow=False)
            pylab.savefig('data/energy-difference' + str(count).zfill(4) + '.png')

            # min/max plots
            pylab.clf()
            x = pylab.arange(0, n + 1, 1)
            plt.plot(x, locmax[x], color='b', label='max')
            plt.plot(x, locmin[x], color='r', label='min')
            pylab.savefig('data/max-min' + str(count).zfill(4) + '.png')

            # mumin/mumax plots
            pylab.clf()
            x = pylab.arange(0, n + 1, 1)
            plt.plot(x, mumax[x], color='b', label='max')
            plt.plot(x, mumin[x], color='r', label='min')
            pylab.savefig('data/mumax-mumin' + str(count).zfill(4) + '.png')

            # relative error plot
            pylab.clf()
            x = pylab.arange(1, n + 1, 1)
            plt.plot(x, rel_e[x], '-', color='black')
            # plt.ylim(energy[x].min()-0.1,energy[x].max()+0.1)
            pylab.savefig('data/relative_error' + str(count).zfill(4) + '.png')
            
            pylab.close('all')

        # control
        if en[n] - en[n - 1] > pos_tol:
            print('Energy is increasing. Stopping.')
            break
        if math.isnan(e1[n] + e2[n] + e3[n]):
            print('Blow-up. Stopping.')
            break

#    # to make npz
#    src_list[count] = phi[:, n, :, :]
#    src_list_slice = src_list[:count + 1]
#    tar_list_slice = tar_list[:count + 1]
#    filename = 'train-' + str(count).zfill(4) + '.npz'  # saves over itself every n to capture the last n...
#    savez_compressed(filename, src_list_slice, tar_list_slice)
#    print('Saved dataset: ', filename)

    count += 1
