#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
import pyntbci

#data_dir = os.path.join(os.path.expanduser("~"), "data", "cvep_covert", "experiment")

wd = r'C:\Users\Radovan\OneDrive\Radboud\Studentships\Jordy Thielen\root'
os.chdir(wd)
data_dir = join(wd, "data")
experiment_dir = join(data_dir, "experiment")
files_dir = join(experiment_dir, 'files')
sourcedata_dir = join(experiment_dir, 'sourcedata')
derivatives_dir = join(join(experiment_dir, 'derivatives'))
os.chdir(wd)
data_dir = experiment_dir

#npz_data = np.load(file_path)
#subjects = ["VPpdia", "VPpdib"]
subjects = ["VPpdia", "VPpdib", "VPpdic", "VPpdid", "VPpdie", "VPpdif", "VPpdig", "VPpdih", "VPpdii", "VPpdij",
            "VPpdik", "VPpdil", "VPpdim", "VPpdin", "VPpdio", "VPpdip", "VPpdiq", "VPpdir", "VPpdis", "VPpdit",
            "VPpdiu", "VPpdiv", "VPpdiw", "VPpdix", "VPpdiy", "VPpdiz", "VPpdiza", "VPpdizb", "VPpdizc"]
session = "S001"
tasks = ["covert"]#, "covert"]

event = "refe"
onset_event = True
encoding_length = 0.5
ensemble = False
lags = None  # np.array([0, 65/60])
n_folds = 4

# Loop participants
accuracy = np.zeros((len(subjects), len(tasks), n_folds))
for i_subject, subject in enumerate(subjects):
    print(f"{subject}", end="\t")

    # Loop tasks
    for i_task, task in enumerate(tasks):
        print(f"{task}: ", end="")

        # Load data
        file_dir = os.path.join(derivatives_dir, 'preprocessed', "cvep", f"sub-{subject}")
        fn = os.path.join(file_dir, f"sub-{subject}_task-{task}_cvep_64_ica.npz")

        tmp = np.load(fn)
        fs = int(tmp["fs"])
        X = tmp["X"]
        y = tmp["y"]
        V = tmp["V"]

        # Cross-validation
        folds = np.repeat(np.arange(n_folds), int(X.shape[0] / n_folds))
        for i_fold in range(n_folds):
            # Split data to train and test set
            X_trn, y_trn = X[folds != i_fold, :, :], y[folds != i_fold]
            X_tst, y_tst = X[folds == i_fold, :, :], y[folds == i_fold]

            # Train classifier
            ecca = pyntbci.classifiers.eCCA(lags=lags, fs=fs, cycle_size=V.shape[1] / fs, ensemble=ensemble,
                                            cca_channels=[26, 27, 28, 29, 63])
            X_trn = X_trn[:, :, :9 * V.shape[1]]  # cut to full cycles
            ecca.fit(X_trn, y_trn)

            # Apply classifier
            yh_tst = ecca.predict(X_tst)

            # Compute accuracy
            accuracy[i_subject, i_task, i_fold] = np.mean(yh_tst == y_tst)

        print(f"{accuracy[i_subject, i_task, :].mean():.3f}", end="\t")
    print()

print(f"Average:\tovert: {accuracy[:, 0, :].mean():.3f}\tcovert: {accuracy[:, 1, :].mean():.3f}")

np.savez(os.path.join(data_dir, "derivatives", "cvep_ecca_64_ica.npz"), accuracy=accuracy)
