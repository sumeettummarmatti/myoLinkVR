import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch

# ==========================
# Filtering Functions
# ==========================

def butter_bandpass_filter(data, lowcut=20, highcut=450, fs=2048, order=4):
    """
    Applies a Butterworth bandpass filter to the data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def apply_notch_filter(data, fs=2048, freq=50, q=30):
    """
    Applies a notch filter to remove power line interference.
    """
    b, a = iirnotch(w0=freq / (0.5 * fs), Q=q)
    return filtfilt(b, a, data, axis=0)

def butter_highpass_filter(data, cutoff=0.1, fs=2048, order=4):
    """
    Applies a high-pass filter to remove DC offset.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, data, axis=0)

def preprocess_signal(data, fs=2048):
    """
    Applies the full preprocessing chain: bandpass -> notch -> highpass.
    Expects data shape: (n_samples, n_channels)
    """
    # 1. Bandpass 20-450Hz
    data = butter_bandpass_filter(data, fs=fs)
    # 2. Notch 50Hz
    data = apply_notch_filter(data, fs=fs)
    # 3. Highpass 0.1Hz
    data = butter_highpass_filter(data, fs=fs)
    return data

# ==========================
# Feature Extraction
# ==========================

# Feature Helpers
def MAV(x): return np.mean(np.abs(x))
def WL(x): return np.sum(np.abs(np.diff(x)))
def RMS(x): return np.sqrt(np.mean(x**2))
def EWL(x, p=0.75): return np.sum(np.abs(x)**p)

def ZC(x, threshold=1e-6):
    return np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) >= threshold))

def SSC(x, threshold=1e-6):
    return np.sum(
        ((x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0) &
        (np.abs(x[2:] - x[:-2]) >= threshold)
    )

def EMAV(x, p=0.75):
    weights = np.where((np.abs(x) > 0.2) & (np.abs(x) < 0.8), p, 0.5)
    return np.mean(np.abs(x) * weights)

def extract_features(data, wavelet='bior3.3', levels=4):
    """
    Extracts features from a window of EMG data using DWT.
    
    Args:
        data: (n_samples, n_channels) array.
        wavelet: Wavelet to use (default 'bior3.3').
        levels: Decomposition levels (default 4).
        
    Returns:
        features: Flat array of shape (n_features,). 
                  Order must match model training:
                  For each channel:
                     For each level (A4, D1, D2, D3, D4):
                        [MAV, WL, ZC, SSC, RMS, EWL, EMAV]
    """
    n_samples, n_channels = data.shape
    features = []
    
    # Define feature functions in the order expected by the model
    # Note: The notebook extracted to CSV then melted. 
    # We need to replicate the exact order.
    # The notebook loop was:
    # 1. Approximation (A4)
    # 2. Details (D1, D2, D3, D4)
    # Inside each level, it iterated channels 0..31
    # BUT wait, the model training usually takes a matrix (n_samples, n_features).
    # The pivot in notebook: `index="file", columns="full_feature"`.
    # `full_feature` = `feature` + "_" + `level` + "_C" + `channel`
    # Example: MAV_A4_C0 (MAV of Approx Level 4, Channel 0)
    #
    # The pivot sorts columns alphabetically by default? No, the code said:
    # `X_df = flat_df.pivot(...)`
    # Pandas `pivot` sorts columns alphabetically!
    # "EMAV_A4_C0", "EMAV_A4_C1", ..., "ZC_D4_C31"
    #
    # This is tricky. If the model relies on alphabetical column order (which Pandas does by default on pivot),
    # we must generate features in that EXACT alphabetical order.
    
    # Let's generate a dictionary first, then sort keys to match Pandas behavior.
    
    feature_dict = {}
    
    # Pre-compute DWT for all channels
    coeffs_per_channel = []
    for ch in range(n_channels):
        # pywt.wavedec returns [cA, cD_levels, cD_levels-1, ..., cD1]
        # BUT the notebook code did:
        # approx = coeffs[0]
        # details = [coeffs[1], coeffs[2], ...] where coeffs[1]=D(level), coeffs[2]=D(level-1)...?
        # WAIT. pywt.wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
        # Notebook code:
        # approx_coeffs.append(coeffs[0]) -> A4
        # for lvl in range(LEVELS): detail_coeffs[lvl].append(coeffs[lvl + 1])
        # So coeffs[1] is D4? or D1?
        # PyWavelets docs: [cA_n, cD_n, cD_n-1, ..., cD_1]
        # Notebook: "for i, c in enumerate(coeffs[1:], 1): label=Detail L{i}"
        # So "Detail L1" corresponds to coeffs[1] which is the HIGHEST level detail (lowest frequency detail after approx)?
        # Actually PyWavelets returns [Approximation, Detail_Level_N, Detail_Level_N-1, ..., Detail_Level_1]
        # So coeffs[1] is Detail Level N (e.g. D4).
        # The notebook loop: `for level in range(1, DWT_LEVELS + 1): file_name = f"detail_level{level}.npy"`
        # And it saves `detail_coeffs[lvl-1]` where lvl is 1..4.
        # `detail_coeffs[0]` comes from `coeffs[1]`.
        # So "D1" in notebook corresponds to `coeffs[1]` which is `cD_N` (D4 in standard terminology usually? or just the coarsest detail).
        # Let's stick to the Notebook's naming convention:
        # Notebook "D1" = coeffs[1]
        # Notebook "D2" = coeffs[2]
        # ...
        
        c = pywt.wavedec(data[:, ch], wavelet=wavelet, level=levels)
        coeffs_per_channel.append(c) # [A4, D(coord1), D(coord2)...]

    # Feature names
    feature_funcs = {
        "MAV": MAV, "WL": WL, "ZC": ZC, "SSC": SSC, 
        "RMS": RMS, "EWL": EWL, "EMAV": EMAV
    }
    
    # We need to construct the dictionary keys exactly as `feature + "_" + level + "_C" + channel`
    # Levels: "A4" and "D1"..."D4"
    
    for ch in range(n_channels):
        c = coeffs_per_channel[ch]
        
        # A4 (Approximation) -> coeffs[0]
        level_name = "A4"
        signal = c[0]
        for fname, func in feature_funcs.items():
            key = f"{fname}_{level_name}_C{ch}"
            feature_dict[key] = func(signal)
            
        # Details
        # Notebook: `for level in range(1, DWT_LEVELS + 1):` -> D1, D2, D3, D4
        # Notebook D1 maps to coeffs[1]
        for i in range(1, levels + 1):
            level_name = f"D{i}"
            signal = c[i] # coeffs[1] is D1 per notebook logic
            for fname, func in feature_funcs.items():
                key = f"{fname}_{level_name}_C{ch}"
                feature_dict[key] = func(signal)
                
    # Now verify sorting. Pandas pivot sorts columns alphabetically.
    # We must return a list of values sorted by their keys.
    sorted_keys = sorted(feature_dict.keys())
    return np.array([feature_dict[k] for k in sorted_keys])
