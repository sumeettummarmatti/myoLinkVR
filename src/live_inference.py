import argparse
import time
import collections
import numpy as np
import serial
import sys
import os

# Ensure src can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import preprocessing
import inference

# ==========================
# CONSTANTS
# ==========================
FS = 2048              # Sampling Frequency expected by model (Hz)
WINDOW_SIZE = 2048     # 1 second window
OVERLAP_SIZE = 1024    # 50% overlap
MODEL_CHANNELS = 32    # Channels expected by model
REAL_CHANNELS = 1      # Channels provided by Arduino (currently)

def get_args():
    parser = argparse.ArgumentParser(description="MyoLinkVR Live Inference")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g., COM3 or /dev/tty.usbmodem...)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--model", type=str, default="../models/svm_lda_pipeline.joblib", help="Path to model file")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode (fake data)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Load Model
    try:
        classifier = inference.ModelWrapper(args.model)
        print(f"✅ Model loaded: {args.model}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Setup Data Buffer
    # We need a rolling buffer of shape (WINDOW_SIZE, MODEL_CHANNELS)
    # Using deque of lists might be slow, let's use numpy rolling buffer concept
    data_buffer = np.zeros((WINDOW_SIZE, MODEL_CHANNELS))
    # Fill with zeros initially
    
    ptr = 0 # Current pointer in buffer

    # 3. Connect to Source
    ser = None
    if not args.simulate:
        try:
            print(f"🔌 Connecting to {args.port}...")
            ser = serial.Serial(args.port, args.baud, timeout=1)
            time.sleep(2)
            print("✅ Connected to Serial!")
            print(f"⚠️ WARNING: Hardware provides {REAL_CHANNELS} channel, but model needs {MODEL_CHANNELS}.")
            print("   Duplicating signal to all 32 channels to prevent crash. PREDICTIONS WILL BE INVALID.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   Tip: Use --simulate to test software without hardware.")
            return
    else:
        print("🧪 RUNNING IN SIMULATION MODE (32 Channels, Random Data)")

    print("🚀 Starting Live Inference... (Ctrl+C to stop)")
    
    try:
        last_pred_time = time.time()
        samples_since_pred = 0
        
        while True:
            # --- Acquire Data Chunk ---
            new_data = None
            
            if args.simulate:
                # Simulate receiving a batch of samples (e.g. 50ms worth) at realistic rate
                samples_to_read = 100 # arbitrary chunk
                # Random EMG-like data (centered at 0, scaled to int16 range approx)
                # Raw data in training was likely int16 0-65536 or centered? 
                # Notebook said: "data_volts = (data - 65536/2) * conversion"
                # Let's clean simulated data to be float directly as preprocessing expects
                # BUT wait, preprocessing.py expects raw input? 
                # Notebook extracted raw -> filter -> feature.
                # Assuming raw floats/ints.
                new_data = np.random.randn(samples_to_read, MODEL_CHANNELS) * 1000 
                time.sleep(samples_to_read / FS) # Sleep to mimic real-time
                
            else:
                # Real Serial Read
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if args.verbose and line:
                        print(f"RAW: {line}")
                        
                    val = None
                    # Format 1: "Emg signals: 123" (Bluetooth)
                    if "Emg signals:" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            val_str = parts[1].strip()
                            if val_str.isdigit():
                                val = float(val_str)
                    
                    # Format 2: "123" (USB Serial)
                    elif line.isdigit():
                        val = float(line)
                        
                    if val is not None:
                        # Duplicate to all channels
                        row = np.full((1, MODEL_CHANNELS), val)
                        new_data = row
                        
                except Exception as e:
                    if args.verbose:
                        print(f"Read Error: {e}")

            # --- Update Buffer ---
            if new_data is not None:
                n = len(new_data)
                
                # Roll buffer and add new data
                # Better: keep a rolling buffer.
                # np.roll is inefficient for streaming.
                # Efficient ring buffer:
                # Check if new data fits at end?
                # Actually, simply rolling array is acceptable for this low bandwidth (2kHz * 32 chans)
                data_buffer = np.roll(data_buffer, -n, axis=0)
                data_buffer[-n:, :] = new_data
                
                samples_since_pred += n
                
            # --- Inference Trigger ---
            # Predict every OVERLAP_SIZE samples (or roughly 0.5s)
            if samples_since_pred >= OVERLAP_SIZE:
                # Extract window
                # Current buffer is the window.
                
                # Preprocess
                # Note: Filtering usually needs state or continuity. 
                # Applying filter to just this window might have edge artifacts.
                # For simplicity here: filter the window. Ideally uses `scipy.signal.lfilter` with `zi`.
                # `preprocessing.preprocess_signal` uses `filtfilt` which is non-causal (needs whole signal). 
                # On a window, `filtfilt` is okay but lag is an issue? `filtfilt` is zero-phase, requires future data.
                # It works on the window, but edge effects might exist. Allowed for now.
                
                clean_data = preprocessing.preprocess_signal(data_buffer, fs=FS)
                
                # Extract features
                # Returns shape (n_features,)
                feats = preprocessing.extract_features(clean_data)
                
                # Predict
                pred_label = classifier.predict(feats)
                
                print(f"🔮 Prediction: {pred_label[0]}")
                
                samples_since_pred = 0
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        if ser: ser.close()

if __name__ == "__main__":
    main()
