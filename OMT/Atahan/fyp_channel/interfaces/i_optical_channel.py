"""
IOpticalChannel - Interface for a composed optical channel.

An IOpticalChannel combines multiple IChannelEffect and propagates a 
signal through them in order.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from fyp_channel.interfaces.i_channel_effect import IChannelEffect


class IOpticalChannel(ABC):

    @abstractmethod
    def propagate(self, signal: np.ndarray, Fs: float) -> np.ndarray:
        """
        Propagate a signal through the channel.
        
        Parameters
        
        signal: np.ndarray    complex optical signal
        Fs: float             sampling frequency [Hz]
        
        Returns
        
        np.ndarray            signal output after all effects
        """
        ...

    @abstractmethod
    def add_effect(self, effect: IChannelEffect) -> None:
        """ Add a channel effect to the channel."""
        ...

    @abstractmethod
    def list_effects(self) -> List[str]:
        """ List the names of the effects currently in the channel."""
        ...
        
