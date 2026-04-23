# Vendored: renzocom/PCIst

- Upstream: https://github.com/renzocom/PCIst
- File: `PCIst/pci_st.py` from the `master` branch
- Fetched: 2026-04-22
- License: GPL-3.0 (see `LICENSE`)

## Citation

> Comolatti R, Pigorini A, Casarotto S, Fecchio M, Faria G, Sarasso S,
> Rosanova M, Gosseries O, Boly M, Bodart O, Ledoux D, Brichant J-F,
> Nobili L, Laureys S, Tononi G, Massimini M, Casali AG (2019).
> *A fast and general method to empirically estimate the complexity of
> brain responses to transcranial and intracranial stimulations.*
> Brain Stimulation, 12(5), 1280-1289.
> https://doi.org/10.1016/j.brs.2019.05.013

## Rules

1. Do **not** modify `pci_st.py`. If anything in it is broken or needs
   adapting, wrap it at the call site (`../../pcist.py`), not here.
2. On refresh, replace `pci_st.py` wholesale with the new upstream copy
   and bump the fetched date above.
3. The wrapper in the repo root (`pcist.py`) is responsible for the
   unit convention we use internally (seconds + `(ch, time)` arrays) and
   for returning the dict keys the rest of the pipeline expects.
