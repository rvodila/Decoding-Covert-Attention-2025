#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jordy Thielen (jordy.thielen@donders.ru.nl)
"""

import os
from os.path import join
import numpy as np
import pyntbci
import matplotlib.pyplot as plt

wd = r'C:\Users\Radovan\OneDrive\Radboud\Studentships\Jordy Thielen\root'
os.chdir(wd)
data_dir = join(wd, "data")
experiment_dir = join(data_dir, "experiment")
files_dir = join(experiment_dir, 'files')
sourcedata_dir = join(experiment_dir, 'sourcedata')
derivatives_dir = join(join(experiment_dir, 'derivatives'))
os.chdir(wd)
data_dir = experiment_dir

subjects = [
    "VPpdia", "VPpdib", "VPpdic", "VPpdid", "VPpdie", "VPpdif", "VPpdig", "VPpdih", "VPpdii", "VPpdij", "VPpdik",
    "VPpdil", "VPpdim", "VPpdin", "VPpdio", "VPpdip", "VPpdiq", "VPpdir", "VPpdis", "VPpdit", "VPpdiu", "VPpdiv",
    "VPpdiw", "VPpdix", "VPpdiy", "VPpdiz", "VPpdiza", "VPpdizb", "VPpdizc"
]
tasks = ["overt", "covert"]

n_folds = 4

capfile = os.path.join(os.path.dirname(pyntbci.__file__), "capfiles", "biosemi64.loc")
fid = open(capfile, "r")
channels = []
for line in fid.readlines():
    channels.append(line.split("\t")[-1].strip())

# Loop participants
accuracy = np.zeros((len(subjects), len(tasks), n_folds))
max_corr = np.zeros((len(subjects), len(tasks), n_folds))
for i_subject, subject in enumerate(subjects):
    print(f"{subject}", end="\t")

    # Loop tasks
    for i_task, task in enumerate(tasks):
        print(f"{task}: ", end="")

        # Load data
        file_dir = os.path.join(derivatives_dir, 'preprocessed', "cvep", f"sub-{subject}")
        fn = os.path.join(file_dir, f"sub-{subject}_task-{task}_cvep_64_noica.npz")
        tmp = np.load(fn)
        fs = int(tmp["fs"])
        X = tmp["X"][:, [channels.index("O1"), channels.index("O2")], :]
        y = tmp["y"]

        # Cross-validation
        folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
        for i_fold in range(n_folds):
            # Split data to train and test set
            X_trn, y_trn = X[folds != i_fold, :, ], y[folds != i_fold]
            X_tst, y_tst = X[folds == i_fold, :, ], y[folds == i_fold]

            n_trials, n_channels, n_samples = X_trn.shape

            # Train classifier
            n_cycles = int(n_samples / fs / 2.1)
            X_trn = X_trn[:, :, :int(n_cycles * 2.1 * fs)]
            X_trn = X_trn.transpose((1, 0, 2))  # channels x trials x samples
            X_trn = X_trn.reshape((n_channels, n_trials * n_cycles, int(2.1 * fs)))  # channels x cycles x samples
            y_trn = np.repeat(y_trn, n_cycles)
            T0 = X_trn[0, y_trn == 0, :].mean(axis=0)  # O1 left
            T1 = X_trn[1, y_trn == 1, :].mean(axis=0)  # O2 right

            # Apply classifier
            rho_0 = pyntbci.utilities.correlation(
                X_tst[:, 0, :],
                np.tile(T0, int(np.ceil(n_samples / T0.shape[0])))[:n_samples])[:, 0]  # O1 left
            rho_1 = pyntbci.utilities.correlation(
                X_tst[:, 1, :],
                np.tile(T1, int(np.ceil(n_samples / T1.shape[0])))[:n_samples])[:, 0]  # O2 right
            yh_tst = np.argmax(np.stack((rho_0, rho_1), axis=1), axis=1)

            # Compute accuracy
            accuracy[i_subject, i_task, i_fold] = np.mean(yh_tst == y_tst)

            # Correlate templates
            rho = np.zeros(T0.shape[0])
            for i in range(T0.shape[0]):
                rho[i] = pyntbci.utilities.correlation(T0, np.roll(T1, i))[0, 0]
            max_corr[i_subject, i_task, i_fold] = rho.argmax() / fs

        print(f"{accuracy[i_subject, i_task, :].mean():.3f}", end="\t")
    print()

print(f"Average:\tovert: {accuracy[:, 0, :].mean():.3f}\tcovert: {accuracy[:, 1, :].mean():.3f}")

print(f"Average max corr: {max_corr.mean():.3f}")
plt.hist(max_corr.flatten())
plt.show()

# np.savez(os.path.join(data_dir, "derivatives", "cvep_ecca.npz"), accuracy=accuracy)
