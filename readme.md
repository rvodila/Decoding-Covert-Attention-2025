
# Decoding Lateral Attention Shifts in a Fixed-Gaze Task Using Two Approaches

## Abstract
A brain–computer interface (BCI) speller typically relies on gaze fixation, which excludes users with impaired oculomotor control, like people living with late-stage amyotrophic lateral sclerosis. We present the first step toward a gaze-independent BCI using the code-modulated visual evoked potential (c-VEP). In our study, participants fixated centrally while covertly attending to one of two bilaterally presented, pseudo-randomly flashing stimuli. From the recorded electroencephalography, we independently analyzed three neural signatures of covert visuospatial attention: (1) the c-VEP, (2) occipital alpha-band lateralization, and (3) the P300 response. From these neural signals, we achieved a grand average classification accuracy of 67\,\% (c-VEP), 88\,\% (alpha), and 98\,\% (P300). This demonstrates the viability of covert spatial attention as a control signal for gaze-independent c-VEP BCIs. 

This project decodes lateral attention shifts using three standalone, but complementory methodologies:

1. **code-modulated VIsual Evoked Potential (c-VEP)**  
   Decoding is performed using template-matching the alpha band envelope.
2. **Alpha-Band lateralization (alpha)**  
   Decoding is performed using lateralization in the alpha band envelope.
3. **P300 ERP**  
   Decoding is based on time-locked responses to target stimuli (i.e., the P300 ERP component).

---

## Noisy Participant Channel Rejection

    Certain EEG channels were rejected due to irreparable noise. 
    Please refer to the rejection map for details on noise patterns of the remaining channels.


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
    2. Remove Bad Channels (per participant)
    3. Apply Band-Pass Filtering:
        Alpha Pipeline: [0.5, 30] Hz
        P300 Pipeline: [0.5, 8] Hz
        CVEP Pipeline: [6, 30] Hz
    4. Epoch the Raw Data
    5. Downsample: From 500 Hz to 120 Hz

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

## CVEP Pipeline