# Decoding Lateral Attention Shifts in a Fixed-Gaze Task Using Three Strategies

## Abstract
A gaze-independent brain–computer interface (BCI) speller could benefit individuals with impaired oculomotor control. We present the first step toward a gaze-independent BCI using the code-modulated visual evoked potential (c-VEP). Participants fixated centrally while covertly attending to one of two bilaterally presented, pseudo-randomly flashing stimuli. We benchmarked three neural signatures of covert spatial attention:

1. **c-VEP**: template-matching via reconvolution CCA  
2. **Alpha-band lateralization**: common spatial patterns on 8–13 Hz envelopes  
3. **P300 ERP**: block-Toeplitz LDA on time-locked epochs  

Grand-average accuracies reached 67 % (c-VEP), 88 % (alpha) and 98 % (P300), demonstrating covert attention as a viable control signal for gaze-independent c-VEP BCIs.

---

## 1. Noisy-Channel Rejection
Certain electrodes were discarded per subject due to irreparable noise.  

````` python
subjects_channel_reject = {
  'VPpdia': ['POz','P2'],
  'VPpdib': ['P2','FC1','CP2'],
  'VPpdic': ['P2'],
  'VPpdid': ['P2'],
  'VPpdie': ['P2'],
  'VPpdif': ['P2','POz'],
  'VPpdig': ['P2'],
  'VPpdih': ['P2','C3'],
  'VPpdii': ['P2'],
  'VPpdij': ['P2','P10'],
  'VPpdik': ['P2'],
  'VPpdil': ['P2'],
  'VPpdim': [],
  'VPpdin': ['P2','POz'],
  'VPpdio': [],
  'VPpdip': [],
  'VPpdiq': [],
  'VPpdir': [],
  'VPpdis': [],
  'VPpdit': [],
  'VPpdiu': ['P2'],
  'VPpdiv': ['P2','PO4','P1','PO3'],
  'VPpdiw': ['P1','P3','POz','Pz','CPz'],
  'VPpdix': [],
  'VPpdiy': [],
  'VPpdiz': ['P1','PO3'],
  'VPpdiza': ['PO4'],
  'VPpdizb': ['Fz'],
  'VPpdizc': ['P1','PO3','POz','Pz','P2','FC2']
}
`````

* See **analysis/raw/Rejection Map/** for a birds-eye view of noisy epochs across channels.
* Timeseries and PSDs are in **analysis/raw/timeseries/** and **analysis/raw/psd/**.

---

## 2. Data Loading & Preprocessing

1. **Load** raw XDF
2. **Reject** bad channels (per subject)
3. **Filter**

   * c-VEP: 6–30 Hz
   * Alpha: 0.5–30 Hz
   * P300: 0.5–8 Hz
4. **Epoch** around stimulus onset
5. **Downsample** 500 Hz → 120 Hz

---

## 3. Experiment Structure

* **Gaze**: fixed central cross
* **Stimuli**: two 3°-diameter circles at ±2.1° eccentricity

  * Background: 126-bit Gold codes at 60 Hz (short/long flashes)
  * Foreground: 4 Hz shape stream (magenta hourglass = target)
* **Trial** (20 s): 80 flashes per side (30 ms each)
* **Task**: attend covertly to cued side and count targets

---

## 4. Data Format

* **EEG** `X`: 80 trials × 80 epochs × n\_channels × time
* **Stimulus codes** `Z`: 80 trials × 80 epochs × 2 sides
* **Labels** `y`: 80 trials (attended side)

---

## 5. c-VEP Pipeline

**Goal:** identify the attended pseudo-random flash stream via reconvolution CCA.

1. **Filter** 6–30 Hz, **epoch** 0–20 s, **downsample** to 120 Hz.
2. **Build** event matrix for each trial:

   * Trial onset
   * Short flashes (16.7 ms)
   * Long flashes (33.3 ms)
3. **Structure matrix**: tile each event over a 300 ms response window.
4. **Calibration (training)**:
   Use recorded trials with known flash codes to run CCA, which learns:
   * a spatial filter that combines channels to best capture the flash response
   * a template waveform that represents the typical brain response

5. **Inference (testing)**
For new EEG data:

   * Apply the spatial filter to get a single response signal
   * Compare that signal to each learned template
   * Pick the template with the highest match—that tells you which side was attended.

---

## 6. Alpha-Band Pipeline

**Goal:** exploit contralateral alpha suppression (8–13 Hz) under covert attention.

1. **Filter** 8–13 Hz, **epoch** 0–20 s, **downsample** to 120 Hz.
2. **Spatial filtering** (CSP): compute six filters—three maximizing and three minimizing variance between left/right attention.
3. **Feature extraction**: Hilbert envelope → log transform → time average → 6-D vector.
4. **Classification**: shrinkage LDA (Ledoit–Wolf) on trial-level features.

---

## 7. P300 ERP Pipeline

**Goal:** detect the infrequent magenta-hourglass ERP on the attended side.

1. **Filter** 0.5–8 Hz, **epoch** 0–20 s, **downsample** to 120 Hz.
2. **Segment** into 80 epochs per trial (−200 ms … +700 ms around each shape).
3. **Spatiotemporal features**: mean amplitude in six windows (50–120, 121–200, …, 531–700 ms) × C channels → `6C`-D.
4. **Epoch-level**: block-Toeplitz LDA for target/non-target classification.
5. **Trial-level**: correlate epoch scores with left/right target sequences; choose side with higher correlation.

---

## 8. Performance Summary

| Strategy             | 5 s    | 10 s   | 15 s   | 20 s   |
| -------------------- | ------ | ------ | ------ | ------ |
| **c-VEP** (with ICA) | 59.9 % | 63.4 % | 65.7 % | 67.1 % |
| **Alpha** (with ICA) | 74.6 % | 82.5 % | 86.8 % | 87.5 % |
| **P300** (with ICA)  | 89.0 % | 95.6 % | 97.6 % | 98.1 % |

> *Note: ICA had minimal impact on all pipelines.*

---

## 9. Discussion & Next Steps

* **P300**: fastest, **alpha**: intermediate, **c-VEP**: slower/plateauing
* **Inter-subject variability** suggests future hybrid decoding of c-VEP + alpha + P300
* **c-VEP improvements**: optimize code sequences, stimulus parameters (size, eccentricity), or invisible stimulation
* **Hybrid architectures** may maximize robustness for diverse neural profiles

---

## 10. References

1. Treder & Blankertz (2010). Covert Attention in ERP-BCI. *Behavioral and Brain Functions*.
2. Treder et al. (2011). Alpha lateralization & covert shifts. *Journal of NeuroEngineering and Rehabilitation*.
3. Narayanan et al. (2024). Pilot gaze-independent c-VEP BCI. *Graz BCI Conference*.
4. Thielen et al. (2021). Reconvolution CCA for c-VEP. *Journal of Neural Engineering*.
5. Sosulski & Tangermann (2022). Block-Toeplitz LDA. *Journal of Neural Engineering*.
