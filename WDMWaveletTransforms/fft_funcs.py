"""helper to make sure available fft functions are consistent across modules depending on install
mkl-fft is faster so it is the default, but numpy fft is probably more commonly installed to it is the fallback
"""
from __future__ import annotations

import numpy as np

__all__ = ['fft', 'ifft', 'irfft', 'rfft']

# Type aliases for signatures (based on numpy 1.20+ stubs, simplified)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


# ----- rfft -----
def rfft(a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None) -> NDArray[np.complex128]: # type: ignore [empty-body]
    ...
# ----- irfft -----
def irfft(a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None) -> NDArray[np.float64]: # type: ignore [empty-body]
    ...
# ----- fft -----
def fft(a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None) -> NDArray[np.complex128]: # type: ignore [empty-body]
    ...
# ----- ifft -----
def ifft(a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None) -> NDArray[np.complex128]: # type: ignore [empty-body]
    ...

try:
    import mkl_fft  # type: ignore[import]

    rfft = mkl_fft.rfft_numpy  # type: ignore[assignment]
    irfft = mkl_fft.irfft_numpy  # type: ignore[assignment]
    fft = mkl_fft.fft  # type: ignore[assignment]
    ifft = mkl_fft.ifft  # type: ignore[assignment]
except ImportError:
    rfft = np.fft.rfft  # type: ignore[no-redef,assignment]
    irfft = np.fft.irfft  # type: ignore[no-redef,assignment]
    fft = np.fft.fft  # type: ignore[no-redef,assignment]
    ifft = np.fft.ifft  # type: ignore[no-redef,assignment]
