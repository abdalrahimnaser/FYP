import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
import logging as logg
import time
from scipy.special import erfc

#OptiCommPy Imports
from optic.models.devices import mzm, photodiode, edfa, iqm, coherentReceiver, pdmCoherentReceiver, basicLaserModel
from optic.models.channels import linearFiberChannel, ssfm
from optic.comm.modulation import modulateGray, grayMapping
from optic.comm.sources import bitSource, symbolSource
from optic.dsp.core import upsample, pulseShape, pnorm, anorm, signalPower, firFilter, decimate, symbolSync, phaseNoise
try:
    from optic.dsp.coreGPU import checkGPU
    if checkGPU():
        from optic.dsp.coreGPU import firFilter
    else:
        from optic.dsp.core import firFilter
except ImportError:
    from optic.dsp.core import firFilter
from optic.utils import parameters, dBm2W, ber2Qfactor
from optic.plot import eyediagram, pconst, plotPSD
try:
    from optic.models.modelsGPU import manakovSSF
except:
    from optic.models.channels import manakovSSF
from optic.dsp.equalization import edc, mimoAdaptEqualizer, ffe
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import fastBERcalc, monteCarloGMI, monteCarloMI, calcEVM, bert
from optic.dsp.clockRecovery import gardnerClockRecovery

logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

# 1. OPTICOMMPY SYSTEM PARAMETERS CONFIGURATION

# General Parameters
SpS = 16                 
Rs = 32e9                
Fs = Rs * SpS            
M = 16                   
nBits = 400000           
rollOff = 0.01           
nFilterTaps = 1024       
mzmScale = 0.5           
Vpi = 2                   
P_launch_dBm = 0         
laserLinewidth = 100e3   

# Symbol Source Parameters
paramSymb = parameters()
paramSymb.nSymbols = int(nBits // np.log2(M))
paramSymb.M = M
paramSymb.constType = "qam"
paramSymb.dist = "uniform"
paramSymb.seed = 123
paramSymb.shapingFactor = 0

constSymb = grayMapping(paramSymb.M, paramSymb.constType)
px = np.ones(paramSymb.M) / paramSymb.M
paramSymb.px = px

# UpSampling and FIR Parameters
paramPulse = parameters()
paramPulse.pulseType = "rrc"
paramPulse.nFilterTaps = nFilterTaps
paramPulse.rollOff = rollOff
paramPulse.SpS = SpS

# IQM Parameters
paramIQM = parameters()
paramIQM.Vpi = Vpi
paramIQM.VbI = -Vpi
paramIQM.VbQ = -Vpi
paramIQM.Vphi = Vpi/2

# Optical Carrier / LO field
sigTx_length = paramSymb.nSymbols * SpS
if laserLinewidth and laserLinewidth > 0:
    phi_pn = phaseNoise(laserLinewidth, sigTx_length, 1 / Fs, seed=123)
    sigLO = np.exp(1j * phi_pn)
else:
    sigLO = np.ones(sigTx_length, dtype=complex) # Fixed bug here

# Optical Channel Parameters
paramCh = parameters()
paramCh.Ltotal = 400       
paramCh.Lspan  = 80        
paramCh.alpha = 0.2        
paramCh.D = 16             
paramCh.gamma = 1.3        
paramCh.Fc = 193.1e12      
paramCh.hz = 0.5           
paramCh.prgsBar = False    
paramCh.Fs = Rs*SpS        
paramCh.amp = 'edfa'       
paramCh.NF = 4.5           

# Receiver Parameters (LO, FE, PD)
FO = -128e6                 
paramLO = parameters()
paramLO.P = 2              
paramLO.lw = 100e3          
paramLO.RIN_var = 0
paramLO.Fs = Fs
paramLO.seed = 789 
paramLO.freqShift = 0 + FO  

paramFE = parameters()
paramFE.Fs = Fs

paramPD = parameters()
paramPD.B = Rs
paramPD.Fs = Fs
paramPD.ideal = True
paramPD.seed = 1011

paramRxPulse = parameters()
paramRxPulse.SpS = SpS
paramRxPulse.nFilterTaps = nFilterTaps
paramRxPulse.rollOff = rollOff
paramRxPulse.pulseType = "rrc"

paramDec = parameters()
paramDec.SpSin  = SpS
paramDec.SpSout = 2

paramEDC = parameters()
paramEDC.L = paramCh.Ltotal
paramEDC.D = paramCh.D
paramEDC.Fc = paramCh.Fc
paramEDC.Rs = Rs
paramEDC.Fs = 2*Rs

paramEq = parameters()
paramEq.nTaps = 35
paramEq.SpS = paramDec.SpSout
paramEq.numIter = 2
paramEq.storeCoeff = False
paramEq.M = M
paramEq.shapingFactor = 0
paramEq.constType = "qam"
paramEq.prgsBar = False
paramEq.alg = ['da-rde','rde'] 
paramEq.mu = [5e-3, 5e-4]

paramCPR = parameters()
paramCPR.alg = 'bps'
paramCPR.M   = M
paramCPR.constType ="qam"
paramCPR.shapingFactor = 0
paramCPR.N   = 25
paramCPR.B   = 64
paramCPR.returnPhases = True
paramCPR.Ts = 1/Rs

# 2. OPTICOMMPY SIMULATION FUNCTION 

def simulate_optical_system(symbTx, paramSymb, paramPulse, paramIQM, sigLO, paramCh, paramLO, paramFE, paramPD, paramRxPulse, paramDec, paramEDC, paramEq, paramCPR):
    # Transmitter
    pulse = pulseShape(paramPulse)
    symbolsUp = upsample(symbTx, SpS)
    sigTx = firFilter(pulse, symbolsUp)
    sigTx = sigTx / np.max(np.abs(sigTx))
    
    u_drive = mzmScale * sigTx
    sigTxo = iqm(sigLO, u_drive, paramIQM)
    
    P_launch_W = dBm2W(P_launch_dBm)
    sigTxo = np.sqrt(P_launch_W) * pnorm(sigTxo)

    # Channel
    sigCh = ssfm(sigTxo, paramCh)

    # Receiver
    paramLO.Ns = len(sigCh)
    sigLO_Rx = basicLaserModel(paramLO)
    sigRxFrontEnd = coherentReceiver(sigCh, sigLO_Rx, paramFE, paramPD)

    rx_pulse = pulseShape(paramRxPulse) 
    sigRxPulseShape = firFilter(rx_pulse, sigRxFrontEnd)
    sigRxDecimation = decimate(sigRxPulseShape, paramDec)
    sigRxCD = edc(sigRxDecimation, paramEDC)
    
    symbRxCD = symbolSync(sigRxCD, symbTx, 2)
    x = pnorm(sigRxCD) 
    d = pnorm(symbRxCD)

    paramEq.L = [int(0.2*d.shape[0]), int(0.8*d.shape[0])]
    y_EQ = mimoAdaptEqualizer(x, paramEq, d)
    y_CPR_1, θ = cpr(y_EQ, paramCPR)

    return y_CPR_1, d

# 3. WIENER-HAMMERSTEIN DPD CORE MODEL

class WienerHammersteinDPD:
    def __init__(self, filter_length=5, ridge=1e-5):
        self.L = filter_length
        self.ridge = ridge
        self.g1 = np.zeros(self.L, dtype=np.complex128)
        self.g1[0] = 1.0 + 0j
        self.g2 = np.zeros(self.L, dtype=np.complex128)
        self.g2[0] = 1.0 + 0j
        self.a = np.array([1.0+0j, 0.0+0j, 0.0+0j], dtype=np.complex128)
        
    def _fir(self, x, h):
        return np.convolve(x, h, mode='full')[:len(x)]
        
    def apply(self, x):
        u = self._fir(x, self.g1)
        w = self.a[0]*u + self.a[1]*u*(np.abs(u)**2) + self.a[2]*u*(np.abs(u)**4)
        return self._fir(w, self.g2)

    def fit_ila(self, pa_in, pa_out):
        def solve_ridge(Phi, d):
            A = Phi.conj().T @ Phi + self.ridge * np.eye(Phi.shape[1])
            b = Phi.conj().T @ d
            return np.linalg.solve(A, b)

        u = self._fir(pa_out, self.g1)
        w = self.a[0]*u + self.a[1]*u*(np.abs(u)**2) + self.a[2]*u*(np.abs(u)**4)
        W = np.zeros((len(w), self.L), dtype=np.complex128)
        for i in range(self.L):
            W[i:, i] = w[:len(w)-i]
        self.g2 = solve_ridge(W, pa_in)

        u = self._fir(pa_out, self.g1)
        Phi = np.column_stack([u, u*(np.abs(u)**2), u*(np.abs(u)**4)])
        Phi_filtered = np.zeros_like(Phi, dtype=np.complex128)
        for col in range(Phi.shape[1]):
            Phi_filtered[:, col] = self._fir(Phi[:, col], self.g2)
        self.a = solve_ridge(Phi_filtered, pa_in)

def calculate_nmse(true_signal, measured_signal):
    error_power = np.mean(np.abs(true_signal - measured_signal)**2)
    signal_power = np.mean(np.abs(true_signal)**2)
    return 10 * np.log10(error_power / signal_power)


# 4. ITERATIVE CONTROL FLOW & BLACKBOX

def opticommpy_blackbox(dpd_tx_symbols):
    y_rx, _ = simulate_optical_system(
        dpd_tx_symbols, paramSymb, paramPulse, paramIQM, sigLO, 
        paramCh, paramLO, paramFE, paramPD, paramRxPulse, paramDec,
        paramEDC, paramEq, paramCPR 
    )
    return y_rx

def iterative_compensation_loop(dpd_model, tx_target, optical_channel, iterations=5, discard=5000):
    print(f"\n--- Starting OptiCommPy Closed-Loop Iterative Compensation ---")
    current_pa_in = tx_target.copy()
    for i in range(iterations):
        current_pa_out = optical_channel(current_pa_in)
        valid_target = tx_target[discard:-discard]
        valid_out = current_pa_out[discard:-discard]
        valid_in = current_pa_in[discard:-discard]
        
        error_db = calculate_nmse(valid_target, valid_out)
        print(f"Iteration {i+1} | Receiver NMSE: {error_db:.4f} dB")
        
        dpd_model.fit_ila(valid_in, valid_out)
        current_pa_in = dpd_model.apply(tx_target)
        
    print("--- Iterative Compensation Complete! ---")
    return current_pa_out

# 5. MAIN EXECUTION

if __name__ == "__main__":
    # 1. Generate pure Symbols
    symbTx = symbolSource(paramSymb)
    
    # 2. Init DPD
    wh_dpd = WienerHammersteinDPD(filter_length=5, ridge=1e-3)
    
    # 3. Run Loop
    y_final_rx = iterative_compensation_loop(
        dpd_model=wh_dpd,
        tx_target=symbTx,
        optical_channel=opticommpy_blackbox,
        iterations=5,
        discard=5000
    )
    
    # 4. Metrics & Plots
    print("\n>>> Final Performance Metrics After DPD <<<")
    discard = 5000
    ind = np.arange(discard, len(symbTx) - discard)
    
    BER, SER, SNR = fastBERcalc(y_final_rx[ind], symbTx[ind], M, 'qam', px=paramSymb.px)
    EVM = calcEVM(y_final_rx[ind], M, 'qam', symbTx[ind])
    Qfactor = ber2Qfactor(BER[0])
    
    print(' Final SER: %.3e'%(SER[0]))
    print(' Final BER: %.3e'%(BER[0]))
    print(' Final SNR: %.3f dB'%(SNR[0]))
    print(' Final EVM: %.3f %%'%(EVM[0]*100))
    print(' Final Qfactor: %.3f'%(Qfactor))
    
    # Plot final Constellation
    pconst(y_final_rx[discard:-discard])