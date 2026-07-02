"""About tab: what PCIst is, what this workbench does, how to read the outputs."""

from __future__ import annotations

import streamlit as st


_TEXT = """
### What this workbench computes

This app runs the **Comolatti 2019 PCIst** (Perturbational Complexity Index
based on State Transitions) on sensor-level TMS-EEG. PCIst summarises the
evoked response by:

1. **SVD dimensionality reduction** on the response-window evoked matrix,
   retaining components up to a cumulative variance threshold (default 99 %)
   and filtering by per-component signal-to-noise ratio.
2. **State-transition quantification** - for each retained component, count
   state transitions in a baseline recurrence matrix and a response
   recurrence matrix across a sweep of distance thresholds; pick the
   threshold that maximises ΔNST = NST_response − k · NST_baseline
   (``k = 1.2`` by default).
3. **Sum** the per-component ΔNST × n_response contributions.

PCIst is computed by the authors' reference implementation
(``renzocom/PCIst``) vendored under ``third_party/pcist_ref/``. Our wrapper
handles unit conversion (seconds ↔ ms) and result packaging only - the
underlying math is unmodified.

### Pipeline

```
BrainVision triple → parse headers/markers → session detection
  → load segment per session
  → cubic-spline TMS artifact interpolation (default [-2, 10] ms)
  → decimate to target fs (default 1000 Hz)
  → automatic bad-channel detection (var > 5× median)
  → common average re-reference
  → bandpass 0.1-45 Hz (zero-phase)
  → epoch (-500, +350) ms, peak-to-peak reject at threshold µV
  → average → evoked
  → PCIst (reference implementation)
```

### Interpretation

- **PCIst is not on a 0-1 scale.** It is a sum across retained SVD components
  of ΔNST × n_response and therefore depends on sampling rate, response
  window, and SNR filtering. Use it for within-study, within-pipeline
  comparisons.
- Classic Casali 2013 thresholds (0.31 / 0.44) were calibrated on
  **source-space** LZ-PCI, not this sensor-space PCIst. Do not apply them
  directly here without separate validation.
- **QC first.** Every session card shows SNR, epoch acceptance, and bad
  channels; the summary table flags sessions with warnings. Treat the
  PCIst value as a relative complexity score and always read the QC
  indicators alongside it.

### References

- Casali AG et al. (2013). *A theoretically based index of consciousness
  independent of sensory processing and behavior.* Sci. Transl. Med.
- **Comolatti R et al. (2019). *A fast and general method to empirically
  estimate the complexity of brain responses to transcranial and intracranial
  stimulations.* Brain Stimulation 12(5):1280-1289.**
- Rogasch NC et al. (2017). *TESA: An open-source toolbox for analyzing
  TMS-EEG data.* NeuroImage.

### Reference implementation

The PCIst math in this workbench is the unmodified ``pci_st.py`` from
<https://github.com/renzocom/PCIst>, vendored in
``third_party/pcist_ref/`` (GPL-3.0, see ``LICENSE`` there).
"""


def render() -> None:
    st.markdown(_TEXT)
