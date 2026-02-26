"""
IChannelEffect - Interface for a single channel impairment.

All physical effects (Chromatic dispersion, EDFA...) implements this
interface, enabling the CompositeChannel to treat all effects.
"""
from abc import ABC, abstractmethod
import numpy as np


class IChannelEffect(ABC):

    @abstractmethod
    def apply(self, signal: np.ndarray, Fs: float) -> np.ndarray:
        """
        Apply the channel effect to a signal.
        
        Parameters
        
        signal: np.ndarray    complex optical signal
        Fs: float             sampling frequency [Hz]
        
        Returns
        
        np.ndarray            signal output after applying the effect
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """ Name of the channel effect."""
        ...
        