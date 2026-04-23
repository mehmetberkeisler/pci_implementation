# TMS-EEG PCIst Workbench

Streamlit and CLI tools for multi-session TMS-EEG PCIst analysis from BrainVision files.
The active pipeline estimates sensor-level PCIst following the state-transition approach of Comolatti et al. (2019), with explicit quality-control warnings for triggers, epochs, SNR, and bad channels.

## Quick Start

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8501 --server.address localhost
```

Open `http://localhost:8501` and upload the matching `.vhdr`, `.vmrk`, and `.eeg` files.

## CLI

```bash
python3 analyze_pci.py /path/to/file.vhdr --gap-seconds 60 --pcist-n-steps 100
```

The repository also contains `pci.py`, a legacy LZ-PCI implementation retained for reference and tests. The Streamlit multi-session workflow uses `analyze_pci.py` and PCIst, not the legacy Casali-style LZ-PCI path.

## Active Method

1. Load BrainVision header, marker, and EEG data.
2. Detect stimulation sessions from periodic marker trains.
3. Accept Response markers as stimulus proxies only when they form a periodic stimulation-like train.
4. Interpolate the TMS artifact window.
5. Downsample to the target processing rate, default 1000 Hz.
6. Remove bad channels, then apply common average reference.
7. Bandpass filter 0.1 to 45 Hz.
8. Extract and reject epochs.
9. Average accepted epochs to an evoked response.
10. Compute PCIst with SVD dimensionality reduction and normalized recurrence-matrix state transitions.

## PCIst Interpretation

PCIst is not the same scale as original Casali 2013 LZ-PCI. Original PCI is a normalized Lempel-Ziv complexity measure on source-reconstructed cortical activation matrices, which is why published values are often discussed near 0 to 1 and why thresholds such as 0.31 or 0.44 appear in that literature.

The active app reports sensor-level PCIst. PCIst sums positive normalized state-transition differences across retained SVD components after multiplying each component contribution by the response-window length. Therefore PCIst is not bounded to 0-1, and the value depends on preprocessing, sampling rate, response window, SVD component selection, SNR filtering, and the threshold scan. Use it mainly for within-study and within-pipeline comparisons unless you have an independently validated threshold for the exact acquisition and processing pipeline.

## References

- Casali AG et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. Science Translational Medicine.
- Comolatti R et al. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. Brain Stimulation.
- Rogasch NC et al. (2017). TESA: An open-source toolbox for analyzing TMS-EEG data. NeuroImage.
