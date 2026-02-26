"""
CDCompensation - IChannelEffect add-on.
"""
import numpy as np
import scipy.constants as const
from numpy.fft import fft, ifft, fftfreq
from fyp_channel.interfaces.i_channel_effect import IChannelEffect


class CDCompensation(IChannelEffect):
    """ 
    CD compensation using direct fft
    
    Parameters
    L: float   fiber span length [km]
    D: float   CD parameter [ps/nm/km]
    Fc: float  carrier frequency [Hz]
    """
    def __init__(
        self,
        L: float = 80.0,
        D: float = 16.0,
        Fc: float = 193.1e12,
    ) -> None:  
        self._L = L
        self._D = D
        self._Fc = Fc

    def apply(self, signal: np.ndarray, Fs: float) -> np.ndarray:
        """ Apply CD compenstion using direct fft."""

        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array.")
        if signal.size == 0:
            raise ValueError("Signal cannot be empty.")
        if Fs <= 0:
            raise ValueError("Sampling frequency must be positive.")
        
        #calculate β2 
        c_kms = const.c / 1e3  # speed of light in km/s
        λ = c_kms / self._Fc  # wavelength in km
        β2 = - (self._D * λ**2) / (2 * np.pi * c_kms)

        input1D = False
        if signal.ndim == 1:
            input1D = True
            signal = signal.reshape(1, -1)

        Nfft = len(signal)
        nModes = signal.shape[1]

        #generate angular frequency vector
        ω = 2 * np.pi * Fs * fftfreq(Nfft)
        ω = ω.reshape(-1, 1)

        #apply CD compensation
        H_comp = np.exp(1j * 0.5 * β2/2 * (ω**2) * self._L)

        compensated = ifft(fft(signal, axis=0) * H_comp, axis=0)

        #return iin same format as input
        if input1D:
            compensated = compensated.flatten()

        return compensated
    
    @property
    def name(self) -> str: 
        return "CD Compensation"
    
    @property
    def L(self) -> float:
        return self._L
    
    @property
    def D(self) -> float:
        return self._D
    
    @property
    def Fc(self) -> float:
        return self._Fc
    
    