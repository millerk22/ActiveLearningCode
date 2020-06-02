import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import re


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

latex_name ={"modelchange_p2":"Probit MC",
         "vopt_p2":"Probit VOpt",
         "modelchange_gr":"GR MC",
         "sopt_gr":"GR SOpt",
         "vopt_gr":"GR VOpt",
         "mbr_gr": "GR MBR",
         "mbr_p2" : "Probit MBR",
         "vopt_p2NA" : "Probit VOpt NA",
         "modelchange_p2NA" :  "Probit MC NA",
         "mbr_p2NA" : "Probit MBR NA",
         "mbr_hf":"HF MBR",
         "vopt_hf": "HF VOpt",
         "sopt_hf":"HF SOpt"}

MARKERS = ['o','*', 'v', '^', '<', '>', '8', 's', 'p', 'h']
COLORS = ['b', 'g', 'r', 'k', 'y', 'purple', 'cyan', 'brown', 'pink', 'orange']



def get_avg_std(filepath, acqs, suffix="t1-g1"):
    ACC = {}
    t = 0
    print(filepath)
    # Compile all the data in the different directories
    for dir in os.listdir(filepath):
        if re.search(suffix, dir):
            data = np.load(filepath + dir + "/acc.npz")
            for acq in data:
                if acq in acqs:
                    if t == 0:
                        ACC[acq] = data[acq][:,np.newaxis]
                    else:
                        ACC[acq] = np.hstack((ACC[acq], data[acq][:, np.newaxis]))

            t += 1

    # Compute average and standard deviation for plots
    avg = {}
    std = {}
    for acq in ACC:
        avg[acq] = np.average(ACC[acq], axis = 1)
        std[acq] = np.std(ACC[acq], axis=1)

    return avg, std


def plot_acc_from_npz(filepath, suffix="t1-g1", acqs=['vopt_gr', 'vopt_p2'], dataset_title="2 Moons", err_bar=False):
    '''
    Plot the Accuracy from the runs contained in the npz file at **filename**

    Assuming that the .npz file
    '''

    print(acqs)
    means, stds = get_avg_std(filepath, acqs, suffix)
    c = 0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for acq in means:
        if not err_bar:
            ax.plot(means[acq], color = COLORS[c], label=latex_name[acq])
        else:
            ax.errorbar(np.arange(1, means[acq].shape[0]+1), means[acq], yerr=stds[acq], label=latex_name[acq], color=COLORS[c]) #fmt='o', color='black',ecolor='lightgray', elinewidth=3, capsize=0)

        c += 1
    ax.set_title(dataset_title + " Accuracy Comparison")
    ax.set_xlabel("Number of labeled points")
    ax.set_ylabel("Accuracy")
    ax.legend()

    return ax




def show_al_choices(X, labels, labeled, n_start=0, acq_name='vopt_gr'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[labels==1, 0], X[labels==1,1], marker='o', c='r', alpha=0.6)
    ax.scatter(X[labels==-1, 0], X[labels==-1,1], marker='x', c='b', alpha=0.6)
    ax.scatter(X[labeled[:n_start],0], X[labeled[:n_start],1], marker='^', c='g', alpha=0.8, s=100, label='Starting Choices')
    ax.scatter(X[labeled[n_start:],0], X[labeled[n_start:],1], marker='*', c='#ffff00', alpha=1.0, s=190, label='Acquisition Choices', edgecolor='k')
    ax.set_title(latex_name[acq_name])
    return ax


def show_many_al_choices(X, labels, filename, acqs, n_start=0):
    LABELED = np.load(filename + "/labeled.npz")
    for acq in acqs:
        if acq not in LABELED:
            print("Did not find data for %s acquisition function, continuing without it..." % acq)
        else:
            print(LABELED[acq].shape)
            ax = show_al_choices(X, labels, LABELED[acq], n_start=n_start, acq_name=acq)
            plt.show()
    return
