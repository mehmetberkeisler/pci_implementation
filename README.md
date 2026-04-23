# TMS-EEG PCIst Workbench

A Streamlit web app for multi-session TMS-EEG analysis using the **PCIst** (Perturbational Complexity Index based on State Transitions) method from Comolatti et al. (2019).

Upload a BrainVision recording (`.vhdr` + `.vmrk` + `.eeg`). The app auto-detects stimulation sessions, runs the full preprocessing pipeline, and reports per-session PCIst with quality-control indicators.

---

## Requirements

- Python **3.9 or later**
- ~500 MB disk space for dependencies
- Your BrainVision EEG files (`.vhdr`, `.vmrk`, `.eeg`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mehmetberkeisler/pci_implementation.git
cd pci_implementation
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py --server.port 8501 --server.address localhost
```

Then open your browser and go to:

```
http://localhost:8501
```

---

## How to Use

1. **Upload your recording** — drag and drop all three BrainVision files (`.vhdr`, `.vmrk`, `.eeg`) into the sidebar uploader at the same time.
2. **Check the preview** — the sidebar instantly shows channel count, sampling rate, duration, and auto-detected sessions.
3. **Adjust parameters** (optional) — expand "Advanced (Comolatti defaults)" in the sidebar to tune epoch rejection threshold, artifact window, target sampling rate, PCIst `k`, etc.
4. **Run analysis** — click the **Run analysis** button in the Analyse tab.
5. **View results** — per-session PCIst values are shown alongside SNR, epoch counts, bad channels, and QC warnings. Plots include the evoked response, GFP, and PCIst component breakdown.

---

## Project Structure

```
pci_implementation/
├── app.py                     # Streamlit entry point
├── analyze_pci.py             # Full analysis pipeline (BrainVision loader, preprocessing, epoching)
├── pcist.py                   # Adapter around the vendored PCIst reference
├── requirements.txt           # Python dependencies
├── ui/                        # Streamlit UI components
│   ├── sidebar.py             # File upload + parameter controls
│   ├── analyze_tab.py         # Run button + progress display
│   ├── results.py             # Per-session result cards
│   ├── plots.py               # Evoked / GFP / PCIst plots
│   ├── about.py               # About tab content
│   ├── state.py               # Session state initialisation
│   └── theme.py               # CSS injection + matplotlib defaults
├── third_party/pcist_ref/     # Vendored Comolatti 2019 reference implementation (GPL-3.0)
└── tests/                     # Pytest test suite
```

---

## Pipeline Steps

1. Load BrainVision header, marker, and binary EEG data.
2. Auto-detect stimulation sessions from inter-stimulus timing gaps.
3. Accept Response markers as TMS triggers when they form a periodic train (common in BrainVision setups).
4. Interpolate the TMS artifact window (default −2 to +10 ms, cubic spline).
5. Downsample to target processing rate (default 1000 Hz).
6. Detect and exclude bad channels, then apply common average reference (CAR).
7. Bandpass filter 0.1–45 Hz (zero-phase FFT).
8. Extract epochs (default −500 to +350 ms), reject by peak-to-peak amplitude.
9. Average accepted epochs into an evoked response.
10. Compute PCIst via SVD dimensionality reduction + normalized state-transition recurrence matrix.

---

## Running Tests

```bash
pytest tests/
```

---

## PCIst vs Original PCI — Important Note

This app reports **sensor-level PCIst**, which is **not** the same scale as the original source-space LZ-PCI (Casali 2013). PCIst values are not bounded to [0, 1] and depend on preprocessing settings, sampling rate, and SVD component selection. Use PCIst values for **within-study comparisons** rather than direct comparison against published LZ-PCI thresholds (e.g., 0.31).

---

## References

- Comolatti R et al. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. *Brain Stimulation*, 12(5):1280–1289.
- Casali AG et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*.
- Rogasch NC et al. (2017). TESA: An open-source toolbox for analyzing TMS-EEG data. *NeuroImage*.

The PCIst math is the **unmodified** authors' reference implementation vendored under `third_party/pcist_ref/` ([renzocom/PCIst](https://github.com/renzocom/PCIst), GPL-3.0).
