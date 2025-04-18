
# Decoding Lateral Attention Shifts in a Fixed-Gaze Task Using Two Approaches

This project decodes lateral attention shifts using two complementary methodologies:

1. **Oscillatory Dynamics (Alpha Band)**  
   Decoding is performed using the alpha band envelope.

2. **P300 Component**  
   Decoding is based on time-locked responses to target stimuli (i.e., the P300 ERP component).

---

## Noisy Participant Channel Rejection

    Certain EEG channels were rejected due to irreparable noise. 
    Please refer to the rejection map for details on noise patterns of the remaining channels.


    subjects_channel_reject = {
        "VPpdib": ["CP2"],
        "VPpdih": ["C3"],
        "VPpdizb": ["Fz"],
        "VPpdizc": ["FC2"]
    }

    Snapshots and PSD Plots:
    You can find broadband time series, ERP-bandpass filtered series, and power spectral density (PSD) plots 
    for each channel per participant in the following folder structure:

    analysis/
    ├── raw/
        ├── timeseries/**
        ├── psd/**
        └── Rejection Map/**

    ERP Visualization:
    Visualize ERP features using the P300 Visualization Notebook.


## Data Loading and Preprocessing

    The following preprocessing steps are executed in the loading script:

    1. Load Raw XDF File
    2. Select Electrodes
    [
        "F7", "F3", "Fz", "F4", "F8", "FC1", "FC2", "FC5",
        "FC6", "FCz", "T7", "C3","Cz", "C4", "T8", "CP1",
        "CP2", "CP5", "CP6", 'CPz',"P7", "P3", "Pz", "P4", 
        "P8", "Oz", "O1", "O2"
    ]

    3. Remove Bad Channels (per participant)
    4. Apply Band-Pass Filtering:
        Alpha Pipeline: [0.5, 30] Hz
        P300 Pipeline: [0.5, 8] Hz
    5. Set Common Average Reference (CAR) (just for p300 loading)
    6. Epoch the Raw Data
    7. Downsample: From 500 Hz to 120 Hz

## Experiment Structure

    Participants maintained a fixed gaze at the center of a screen while stimuli were presented simultaneously on the left and right sides. The details of the experimental design are as follows:

        Stimulus Details:
            Stimuli on each side can be either targets or non-targets.
            Participants were instructed to attend to one side and count the number of stimuli presented.
        Trial Structure:
            Duration: 20 seconds per trial.
            Stimulus Presentation: 80 stimuli per side per trial (each stimulus lasts 30 ms).

## Data Format

    EEG Data X:
    80 Trials x 80 Epochs x n_channels x time

    Stimulus Coding Array z:
    80 Trials x 80 Epochs x 2
    (Each trial contains a stimulus code for each side)

    Attended Side Label y:
    80 Trials x 1
    (Indicates the side on which the participant was instructed to attend)

## P300 Pipeline

    The P300 pipeline uses ERP components to differentiate between target and non-target presentations.
    Key Points

        ERP Components:
        Target stimuli elicit an ERP complex featuring prominent P100 and P300 components.

        Decoding Strategy:
            Epoch-level Decoding:
            Apply Linear Discriminant Analysis (LDA) on epochs time-locked to stimulus onset.
            Trial-level Inference:
            Correlate LDA classification outcomes with the event matrix for both sides.
            The side with the highest correlation (i.e., where target was both presented and decoded) is identified as the attended side.

        Performance Assessment:
            Subject-resolved decoding with full trial integration.
            A decoding curve is generated by varying the number of stimulus presentations included in the correlation-based decision rule.

## Alpha Pipeline

    The alpha pipeline is based on the neural marker of alpha band suppression following visual attention.
    Key Points

        Neural Basis:
            Stimuli in the left visual field are processed contralaterally (right visual cortex) and vice versa.
            Attending to a stimulus results in alpha band suppression in the contralateral hemisphere.

    Decoding Strategy

        Channel Selection:
            Method 1: Use a simple sub-selection of electrodes (e.g., occipital electrodes).
            Method 2: Apply Common Spatial Patterns (CSP) to derive virtual channels that maximize variance between conditions.
        Feature Extraction:
            Compute the log-mean alpha band envelope using the Hilbert transform.
            Use the computed envelope as a feature input to the LDA classifier.
        Performance Assessment:
            Subject-resolved decoding performance.
            A decoding curve is generated by varying the time window used to compute the feature vector.