import numpy as np
from optic.models.devices import mzm, photodiode
from optic.models.channels import linearFiberChannel
from optic.comm.sources import bitSource
from optic.comm.modulation import modulateGray
from optic.comm.metrics import bert
from optic.dsp.core import firFilter, pulseShape, upsample, anorm
from optic.utils import parameters, dBm2W
from scipy.special import erfcinv


# =============================================================================
# Volterra/GMP DPD (Keras-like API): predict(x_input) + fit_from_pair(x_input, y_received)
# =============================================================================

def _to_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)

def _to_complex_1d(x: np.ndarray) -> np.ndarray:
    x = _to_1d(x)
    if np.iscomplexobj(x):
        return x.astype(np.complex128, copy=False)
    return x.astype(np.float64, copy=False).astype(np.complex128)

def _odd_orders(P: int) -> np.ndarray:
    return np.arange(1, P + 1, 2, dtype=int)

def _rms(x: np.ndarray) -> float:
    x = _to_complex_1d(x)
    return float(np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12))

def nmse_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref = _to_complex_1d(ref)
    est = _to_complex_1d(est)
    num = np.mean(np.abs(est - ref) ** 2) + 1e-12
    den = np.mean(np.abs(ref) ** 2) + 1e-12
    return 10.0 * np.log10(num / den)

def compute_metrics(symbTx, I_Rx, bitsTx, BER):
    """
    Compute SER, BER, SNR, EVM, Q-factor
    """

    # decision (PAM)
    symbRx = np.sign(I_Rx)
    symbRx[symbRx == 0] = 1

    # SER
    SER = np.mean(symbRx != symbTx)

    # noise
    noise = I_Rx - symbTx

    # SNR
    signal_power = np.mean(symbTx**2)
    noise_power = np.mean(noise**2)
    SNR = 10*np.log10(signal_power/noise_power)

    # EVM
    EVM = np.sqrt(noise_power / signal_power) * 100

    # Q factor (from BER)
    Q = np.sqrt(2) * erfcinv(2*BER)

    return {
        "SER": SER,
        "BER": BER,
        "SNR": SNR,
        "EVM": EVM,
        "Q": Q
    }

def build_gmp_matrix(x: np.ndarray, P: int, M: int, L: int) -> np.ndarray:
    """
    GMP basis:
      Main MP terms:
        x[n-m] * |x[n-m]|^(p-1), p odd, m=0..M
      Cross-envelope terms:
        x[n-m] * |x[n-m-l]|^(p-1), p odd, m=0..M, l=1..L
    """
    x = _to_complex_1d(x)
    N = len(x)
    orders = _odd_orders(P)

    # Main MP
    xpad_mp = np.concatenate([np.zeros(M, dtype=np.complex128), x])
    cols = []
    for p in orders:
        for m in range(M + 1):
            xm = xpad_mp[M - m : M - m + N]
            cols.append(xm * (np.abs(xm) ** (p - 1)))
    Phi_mp = np.stack(cols, axis=1) if cols else np.zeros((N, 0), dtype=np.complex128)

    # Cross terms
    if L <= 0:
        return Phi_mp

    pad = M + L
    xpad = np.concatenate([np.zeros(pad, dtype=np.complex128), x])
    cols = []
    for p in orders:
        for m in range(M + 1):
            x_main = xpad[pad - m : pad - m + N]
            for l in range(1, L + 1):
                x_env = xpad[pad - m - l : pad - m - l + N]
                cols.append(x_main * (np.abs(x_env) ** (p - 1)))
    Phi_cross = np.stack(cols, axis=1) if cols else np.zeros((N, 0), dtype=np.complex128)

    return np.concatenate([Phi_mp, Phi_cross], axis=1)


class VolterraDPD:
    """
    Volterra/GMP DPD with a NN-like interface:
      - predict(x_input): x_input shape (batch, 8192, 1)
      - fit_from_pair(x_input, y_received): trains postdistorter x ≈ f(y) using ridge LS
    """

    def __init__(self, P=7, M=5, L=2, ridge=1e-5, seq_len=8192, normalize=True):
        self.P = int(P)
        self.M = int(M)
        self.L = int(L)
        self.ridge = float(ridge)
        self.seq_len = int(seq_len)
        self.normalize = bool(normalize)

        self.w = None
        self.scale_x = 1.0
        self.scale_y = 1.0

    def _predict_1d(self, x_1d: np.ndarray) -> np.ndarray:
        x_1d = _to_complex_1d(x_1d)

        # If not trained yet -> identity
        if self.w is None:
            return x_1d

        if self.normalize:
            x_n = x_1d / (self.scale_x + 1e-12)
        else:
            x_n = x_1d

        Phi = build_gmp_matrix(x_n, self.P, self.M, self.L)
        z_n = Phi @ self.w

        if self.normalize:
            return z_n * self.scale_x
        return z_n

    def predict(self, x_input: np.ndarray, verbose: int = 0) -> np.ndarray:
        x = np.asarray(x_input)

        # Support: (batch, seq, 1)
        if x.ndim == 3 and x.shape[2] == 1:
            B, T, _ = x.shape
            out = np.zeros((B, T, 1), dtype=np.complex128)
            for b in range(B):
                out[b, :, 0] = self._predict_1d(x[b, :, 0])
            return out

        # Support: (seq,) or (seq,1)
        if x.ndim == 1:
            return self._predict_1d(x).reshape(-1)
        if x.ndim == 2 and x.shape[1] == 1:
            return self._predict_1d(x[:, 0]).reshape(-1, 1)

        raise ValueError(f"Unsupported x_input shape: {x.shape}")

    def fit_from_pair(self, x_input: np.ndarray, y_received: np.ndarray, verbose: int = 0) -> dict:
        """
        Train postdistorter: x ≈ f(y)
        x_input and y_received should represent aligned sequences.
        """
        x = _to_complex_1d(x_input)
        y = _to_complex_1d(y_received)
        if len(x) != len(y):
            raise ValueError(f"Length mismatch: len(x)={len(x)} vs len(y)={len(y)}")

        if self.normalize:
            self.scale_x = _rms(x)
            self.scale_y = _rms(y)
            x_n = x / (self.scale_x + 1e-12)
            y_n = y / (self.scale_y + 1e-12)
        else:
            x_n = x
            y_n = y

        Phi = build_gmp_matrix(y_n, self.P, self.M, self.L)
        N, K = Phi.shape
        if K == 0:
            raise RuntimeError("Empty regression matrix. Check P/M/L.")

        lam = self.ridge
        A = Phi.conj().T @ Phi + lam * np.eye(K, dtype=np.complex128)
        b = Phi.conj().T @ x_n
        self.w = np.linalg.solve(A, b)

        x_hat = Phi @ self.w
        return {
            "coefficients": int(K),
            "train_nmse_db": float(nmse_db(x_n, x_hat)),
        }


# =============================================================================
# Your original perf_sim, with SAME parameters + SAME calling method
# =============================================================================

def perf_sim(dpd_model: VolterraDPD, DPD_FLAG=False, random_seed=123):
    # simulation parameters (same as your code)
    SpS = 16
    M = 2
    Rs = 32e9
    Fs = SpS * Rs
    Pi_dBm = 3
    Pi = dBm2W(Pi_dBm)

    # Bit source parameters
    paramBits = parameters()
    paramBits.nBits = 2**18
    paramBits.mode = 'random'
    paramBits.seed = random_seed

    # pulse shaping parameters
    paramPulse = parameters()
    paramPulse.pulseType = 'nrz'
    paramPulse.SpS = SpS

    # MZM parameters
    paramMZM = parameters()
    paramMZM.Vpi = 2
    paramMZM.Vb = -paramMZM.Vpi / 2

    # linear fiber optical channel parameters
    paramCh = parameters()
    paramCh.L = 100
    paramCh.alpha = 0.2
    paramCh.D = 16
    paramCh.Fc = 193.1e12
    paramCh.Fs = Fs

    # photodiode parameters
    paramPD = parameters()
    paramPD.ideal = False
    paramPD.B = Rs
    paramPD.Fs = Fs
    paramPD.seed = 456

    # generate pseudo-random bit sequence
    bitsTx = bitSource(paramBits)

    # generate 2-PAM modulated symbol sequence
    symbTx = modulateGray(bitsTx, M, "pam")  # real-valued symbols

    # DPD calling method (same as your code)
    x_input = symbTx.reshape(-1, 8192, 1)   # (batch, 8192, 1)
    z_dpd = dpd_model.predict(x_input, verbose=0)
    z_signal = z_dpd.flatten().real         # PAM uses real drive

    # upsampling
    if DPD_FLAG:
        symbolsUp = upsample(z_signal, SpS)
    else:
        symbolsUp = upsample(symbTx, SpS)

    # pulse shaping
    pulse = pulseShape(paramPulse)
    sigTx = firFilter(pulse, symbolsUp)
    sigTx = anorm(sigTx)  # normalize to 1 Vpp

    # optical modulation
    Ai = np.sqrt(Pi)
    sigTxo = mzm(Ai, sigTx, paramMZM)

    # linear fiber channel model
    sigCh = linearFiberChannel(sigTxo, paramCh)

    # noisy PD
    I_Rx = photodiode(sigCh, paramPD)

    # capture samples (same as your code)
    I_Rx = I_Rx[0::SpS]
    

    # calculate BER and Q-factor
    BER, Q = bert(I_Rx, bitsTx)
    metrics = compute_metrics(symbTx, I_Rx, bitsTx, BER)

    return {
        "SER": metrics["SER"],
        "BER": metrics["BER"],
        "SNR": metrics["SNR"],
        "EVM": metrics["EVM"],
        "Q": metrics["Q"],
        "bitsTx": bitsTx,
        "symbTx": symbTx,
        "I_Rx": I_Rx,
        "x_input": x_input,
    }


# =============================================================================
# Training procedure (Volterra ILA-style)
# =============================================================================

def train_volterra_dpd(num_iters=3, seed0=123):
    """
    Training idea for Volterra DPD in your setup:

    We want a mapping that compensates the chain as seen at the sampled PD output.
    We train a postdistorter:
        symbTx ≈ f(I_Rx)
    Then use the same f(.) as predistorter:
        z = f(symbTx)

    This is ILA-like, but using your exact data path and sequence shape.
    """

    dpd_model = VolterraDPD(P=7, M=5, L=2, ridge=1e-5, seq_len=8192, normalize=True)

    for k in range(num_iters):
        seed = seed0 + 100 * k

        # 1) Run baseline to collect training pair (symbTx, I_Rx)
        out = perf_sim(dpd_model, DPD_FLAG=False, random_seed=seed)

        symbTx = out["symbTx"]
        I_Rx = out["I_Rx"]

        # 2) Reshape to match the notebook-style input (batch,8192,1)
        x_train = symbTx.reshape(-1, 8192, 1)
        y_train = I_Rx.reshape(-1, 8192, 1)

        # 3) Fit postdistorter x ≈ f(y)
        info = dpd_model.fit_from_pair(x_train, y_train, verbose=0)

        print(f"[Iter {k+1}/{num_iters}] trained coeffs={info['coefficients']}  train_nmse_db={info['train_nmse_db']:.2f}")

        # 4) Evaluate DPD-enabled performance on a new seed
        test = perf_sim(dpd_model, DPD_FLAG=True, random_seed=seed + 7)
        print(f"           Test with DPD: Q={test['Q']:.2f}  BER={test['BER']:.2e}")

    return dpd_model


if __name__ == "__main__":
    # Train DPD
    dpd = train_volterra_dpd(num_iters=3, seed0=123)

    # Final evaluation
    final = perf_sim(dpd, DPD_FLAG=True, random_seed=345)
    print("\nFinal DPD performance metrics:")
    print(f"Final SER: {final['SER']:.3e}")
    print(f"Final BER: {final['BER']:.3e}")
    print(f"Final SNR: {final['SNR']:.3f} dB")
    print(f"Final EVM: {final['EVM']:.3f} %")
    print(f"Final Qfactor: {final['Q']:.3f}")

    

