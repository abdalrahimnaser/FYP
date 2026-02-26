"""
CompositeChannel - Whole channel with the IChannelEffect add-ons.
"""
from __future__ import annotations
from typing import List
import numpy as np
from fyp_channel.interfaces.i_optical_channel import IOpticalChannel
from fyp_channel.interfaces.i_channel_effect import IChannelEffect


class CompositeChannel(IOpticalChannel):

    def __init__(self) -> None:
        self._effects: List[IChannelEffect] = []

    def propagate(self, signal: np.ndarray, Fs: float) -> np.ndarray:
        """ Propagate the signal through all channel effects."""

        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array.")
        if signal.size == 0:
            raise ValueError("Signal cannot be empty.")
        if Fs <= 0:
            raise ValueError("Sampling frequency must be positive.")
        
        result = signal.copy()
        for effect in self._effects:
            result = effect.apply(result, Fs)
        return result

    def add_effect(self, effect: IChannelEffect) -> None:
        """ Add a channel effect to the channel."""

        if not isinstance(effect, IChannelEffect):
            raise TypeError("An IChannelEffect is needed.")
        
        self._effects.append(effect)

    def list_effects(self) -> List[str]:
        """ List the names of the effects currently in the channel."""
        
        return [effect.name for effect in self._effects]
