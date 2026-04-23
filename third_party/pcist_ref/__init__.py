"""Vendored renzocom/PCIst reference implementation.

Source: https://github.com/renzocom/PCIst (master)
License: GPL-3.0 (see LICENSE in this directory)

This package exposes the authors' canonical PCIst implementation so that
our pipeline calls it unmodified. Do NOT edit pci_st.py — any local
wrapping or unit conversion belongs in ../../pcist.py.
"""

from .pci_st import calc_PCIst  # noqa: F401

__all__ = ["calc_PCIst"]
