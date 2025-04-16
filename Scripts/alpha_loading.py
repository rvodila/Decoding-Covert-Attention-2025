#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RV
"""

import os
import numpy as np
import mne
from mnelab.io.xdf import read_raw_xdf as read_raw
import pyxdf
from os.path import join
import warnings
import joblib
import pandas as pd
warnings.filterwarnings("ignore")
# Set MNE log level
mne.set_log_level('warning')

# Define directories
wd = r'C:\Users\Radovan\OneDrive\Radboud\Studentships\Jordy Thielen\root'
os.chdir(wd)
data_dir = join(wd, "data")
experiment_dir = join(wd, "data", "experiment")
files_dir = join(experiment_dir, 'files')
sourcedata_dir = join(experiment_dir, 'sourcedata')
derivatives_dir = join(join(experiment_dir, 'derivatives'))
ica_dir = join(data_dir, 'analysis', 'ICA')
# EEG channel mapping
channel_dict = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
    'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4',
    'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
    'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2',
    'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
}
all_channels = [*channel_dict.values()]
# Tasks and runs
tasks = ["overt", "covert"]
runs = {"overt": ["001"], "covert": ["001", "002", "003", "004"]}


# Load electrode montage
loc_file = os.path.join(files_dir, "biosemi64.loc")
montage = mne.channels.read_custom_montage(loc_file)

# Load visual stimulus codes
fs = 120  # target EEG (down)sampling frequency in Hz
pr = 60  # stimulus presentation rate in Hz
V = np.load(os.path.join(files_dir, "mgold_61_6521.npz"))["codes"]
V = np.repeat(V, int(fs / pr), axis=1).astype("uint8")

# Subjects and bad channel rejection mapping
subjects = ["VPpdia", "VPpdib", "VPpdic", "VPpdid", "VPpdie", "VPpdif", "VPpdig", "VPpdih", "VPpdii", "VPpdij",
            "VPpdik", "VPpdil", "VPpdim", "VPpdin", "VPpdio", "VPpdip", "VPpdiq", "VPpdir", "VPpdis", "VPpdit",
            "VPpdiu", "VPpdiv", "VPpdiw", "VPpdix", "VPpdiy", "VPpdiz", "VPpdiza", "VPpdizb", "VPpdizc"]

subjects_channel_reject = {'VPpdia': ['POz', 'P2'],
                            'VPpdib': ['P2', 'FC1', 'CP2'],
                            'VPpdic': ['P2'], 
                            'VPpdid': ['P2'], 'VPpdie': ['P2'], 
                            'VPpdif': ['P2', 'POz'], 'VPpdig': ['P2'], 
                            'VPpdih': ['P2', 'C3'], 
                            'VPpdii': ['P2'], 
                            'VPpdij': ['P2', 'P10'], 
                            'VPpdik': ['P2'], 
                            'VPpdil': ['P2'], 
                            'VPpdim': [], 'VPpdin': ['P2', 'POz'], 
                            'VPpdio': [], 'VPpdip': [], 'VPpdiq': [], 
                            'VPpdir': [], 'VPpdis': [], 'VPpdit': [], 
                            'VPpdiu': ['P2'], 'VPpdiv': ['P2', 'PO4', 'P1', 'PO3'], 
                            'VPpdiw': ['P1', 'P3', 'POz', 'Pz', 'CPz'], 
                            'VPpdix': [], 'VPpdiy': [], 'VPpdiz': ['P1', 'PO3'], 
                            'VPpdiza': ['PO4'], 'VPpdizb': ['Fz'], 
                            'VPpdizc': ['P1', 'PO3', 'POz', 'Pz', 'P2', 'FC2']}

# Define sub-set of electrodes
picks_hubner = [
    "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "FC5", "FC6", "FCz", "T7", "C3",
    "Cz", "C4", "T8", "CP1", "CP2", "CP5", "CP6", 'CPz',
    "P7", "P3", "Pz", "P4", "P8", "Oz", "O1", "O2"
]


bandpass = (0.5, 30)  # bandpass with low and high cutoff in Hz
tmin = -0.5  # trial start in seconds
tmax = 20.5  # trial duration in seconds


# Loop subjects
for i_subject, subject in enumerate(subjects):
    print("-" * 25)
    print(f"{subject}")
    
    # Load ICA solution
    ica_path = os.path.join(ica_dir, 'ica', f"sub-{subject}", f"{subject}_ICA_.joblib")
    ica = joblib.load(ica_path)

    # Load the rejected ICA components from CSV
    deleted_components_path = os.path.join(ica_dir, 'ica', f"sub-{subject}", f"sub-{subject}_deleted_eye_components.csv")
    if os.path.exists(deleted_components_path):
        deleted_components = pd.read_csv(deleted_components_path, header=None).to_numpy().flatten()

        # Filter out non-numeric entries (headers)
        deleted_components = [comp for comp in deleted_components if str(comp).isdigit()]

        # Convert the valid components to integers and exclude them
        deleted_components = list(map(int, deleted_components))

        # Ensure that the excluded components are within the range of ICA components
        print(deleted_components)

        ica.exclude = deleted_components
        print(f"Removed components: {ica.exclude}")
    else:
        print("Warning: No deleted components file found!")

    # Loop tasks
    for i_task, task in enumerate(tasks):
        print(f"\t{task}")

        X = []
        y = []
        z = []

        # Loop runs
        for i_run, run in enumerate(runs[task]):

            # Run XDF file name
            fn = os.path.join(sourcedata_dir, f"sub-{subject}", "ses-S001", "eeg",
                              f"sub-{subject}_ses-S001_task-{task}_run-{run}_eeg.xdf")

            # Read EEG
            streams = pyxdf.resolve_streams(fn)
            names = [stream["name"] for stream in streams]
            stream_id = streams[names.index("BioSemi")]["stream_id"]
            raw = read_raw(fn, stream_ids=[stream_id], verbose=False)
            raw.rename_channels(channel_dict)
            raw.set_montage(montage)
            
            # De-meaning
            raw._data[0, :] -= np.median(raw._data[0, :])
            raw._data[0, :] = raw._data[0, :] > 0
            raw._data[0, :] = np.logical_and(raw._data[0, :], np.roll(raw._data[0, :], -1))
            events = mne.find_events(raw, stim_channel="Trig1", verbose=False)

            # If hardware marker missing, use LSL
            if events.shape[0] == 0:
                print(f"\t\tFound {events.shape[0]:d} events in trigger channel.")
                streams = pyxdf.load_xdf(fn)[0]
                names = [stream["info"]["name"][0] for stream in streams]

                stream = streams[names.index("KeyboardMarkerStream")]
                t_mrk = [t for t, mrk in zip(stream["time_stamps"], stream["time_series"])
                         if mrk[2] == "start_stimulus"]

                stream = streams[names.index("BioSemi")]
                t_eeg = stream["time_stamps"]

                events = np.zeros((len(t_mrk), 3), dtype=events.dtype)
                events[:, 0] = np.array([np.argmin(np.abs(t_eeg - t)) for t in t_mrk])
                print(f"\t\tFound {events.shape[0]:d} events in marker stream.")

            # pick sub-set (Huebner)
            raw.pick(all_channels)
            
            # Reject bad channel
            if subject in subjects_channel_reject:
                channels_to_drop = subjects_channel_reject[subject]
                #raw_ch_names = [str(ch) for ch in raw.info['ch_names']]
                 # Ensure that only existing channels are dropped
                channels_to_drop = [ch for ch in channels_to_drop if ch in raw.info['ch_names']]
                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)
            
            # ICA REJECTION

            # Apply ICA
            
            #print(f'pre-ica rank: {mne.compute_rank(raw)}')
            raw = ica.apply(raw)
            #print(f'post-ica rank: {mne.compute_rank(raw)}')
            # Spectral bandpass filter
            raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
            
            epo = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, baseline=None,
                             preload=True, verbose=False)
            
     
            # Resampling
            # N.B. Downsampling is done after slicing to maintain accurate stimulus timing
            epo = epo.resample(sfreq=fs, verbose=False)

            # Read labels
            streams = pyxdf.load_xdf(fn)[0]
            names = [stream["info"]["name"][0] for stream in streams]
            marker_stream = streams[names.index("KeyboardMarkerStream")]
            cued_side = np.array([marker[3].lower().strip('""') == "right"
                                  for marker in marker_stream["time_series"]
                                  if marker[2] == "cued_side"])
            left_target = np.array([x[3].split(";")[0].split("=")[1] == "hour_glass"
                                    for x in marker_stream["time_series"]
                                    if x[2] == "left_shape_stim"]).reshape((cued_side.size, -1))
            right_target = np.array([x[3].split(";")[0].split("=")[1] == "hour_glass"
                                     for x in marker_stream["time_series"]
                                     if x[2] == "right_shape_stim"]).reshape((cued_side.size, -1))
            targets = np.stack((left_target, right_target), axis=2)

            # Extract data
            X.append(epo.get_data(copy=True, verbose=False))
            y.append(cued_side)
            z.append(targets)

        # Stack data
        X = np.concatenate(X, axis=0).astype("float32")
        y = np.concatenate(y, axis=0).astype("uint8")
        z = np.concatenate(z, axis=0).astype("uint8")
        
        # Save data
        save_dir = os.path.join(derivatives_dir, "preprocessed", "alpha", f"sub-{subject}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(os.path.join(save_dir, f"sub-{subject}_task-{task}_alpha_64_ica.npz"), X=X, y=y, z=z, V=V, fs=fs)
