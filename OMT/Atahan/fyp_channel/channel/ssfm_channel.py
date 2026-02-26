"""
SSFMChannel - IChannelEffect add-on.
"""
import numpy as np
from optic.models.channels import ssfm
from optic.utils import parameters
from fyp_channel.interfaces.i_channel_effect import IChannelEffect


class SSFMChannel(IChannelEffect):
    """
    Single polarization nonlinear fiber channel(SSFM).
    
    Parameters
    Ltotatl: float    total fiber length [km]
    Lspan: float      length of each SSFM step [km]
    hz: float         step size for split-step method
    alpha: float      fiber attenuation [dB/km]
    D: float          CD parameter [ps/nm/km]
    gamme: float      fiber nonlinear coefficient [1/W/km]
    Fc: float         carrier frequency [Hz]
    amp: str or None  amplifiaction mode: "edfa", "ideal" or None
    NF: float         EDFA noise figure [dB]
    seed: int or None random seed for ASE noise
    prgsBar: bool     progress bar during propagation
    """
    def __init__(
        self,
        Ltotal: float = 400.0,
        Lspan: float = 80.0,
        hz: float = 0.5,
        alpha: float = 0.2,
        D: float = 16.0,
        gamma: float = 1.3,
        Fc: float = 193.1e12,
        amp: str | None = "edfa",
        NF: float = 4.5,
        seed: int | None = None,
        prgsBar: bool = False,
    ) -> None:
        if Ltotal <= 0:
            raise ValueError("Ltotal must be positive.")
        if Lspan <= 0:
            raise ValueError("Lspan must be positive.")
        if hz <= 0:
            raise ValueError("step size hz must be positive.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        
        self._Ltotal = Ltotal
        self._Lspan = Lspan
        self._hz = hz
        self._alpha = alpha
        self._D = D
        self._gamma = gamma
        self._Fc = Fc
        self._amp = amp
        self._NF = NF
        self._seed = seed
        self._prgsBar = prgsBar

    def apply(self, signal: np.ndarray, Fs: float) -> np.ndarray:
        """Propagate signal through the channel."""
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array.")
        if signal.size == 0:
            raise ValueError("Signal cannot be empty.")
        if Fs <= 0:
            raise ValueError("Sampling frequency must be positive.")
        
        param = parameters()
        param.Ltotal = self._Ltotal
        param.Lspan = self._Lspan
        param.hz = self._hz
        param.alpha = self._alpha
        param.D = self._D
        param.gamma = self._gamma
        param.Fc = self._Fc
        param.Fs = Fs
        param.amp = self._amp
        param.NF = self._NF
        param.seed = self._seed
        param.prgsBar = self._prgsBar
        return ssfm(signal, param)
    
    @property
    def name(self) -> str:
        return "SSFM Channel"
    
    @property
    def Ltotal(self) -> float:
        return self._Ltotal
    
    @property
    def Lspan(self) -> float:
        return self._Lspan
    
    @property
    def n_spans(self) -> int:
        return int(np.floor(self._Ltotal / self._Lspan))
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @property
    def D(self) -> float:
        return self._D
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    
    
    