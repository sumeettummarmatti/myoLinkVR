# MyoLink VR

MyoLink VR is a project focused on EMG-based gesture recognition for Virtual Reality interaction. This repository contains the firmware, model training notebooks, and the live inference pipeline.

##  Live Inference Setup

We have implemented a real-time classification pipeline that reads EMG data, processes it using Discrete Wavelet Transform (DWT), and predicts gestures using trained machine learning models.

###  Critical Hardware Note
The current trained models require **32 channels** of EMG data.
The current Arduino firmware provides **1 channel**.
* **Simulation Mode**: Works perfectly (uses fake 32-channel data).
* **Live Mode**: The script currently **duplicates** the single channel 32 times to prevent the software from crashing. **Predictions in this mode will be invalid** until hardware is upgraded to 32 channels.

###  Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

###  Usage

#### 1. Simulation Mode (Testing)
Run this to verify the software pipeline without hardware. It generates synthetic logic 32-channel data.
```bash
python3 src/live_inference.py --simulate --model models/svm_lda_pipeline.joblib
```

#### 2. Live Hardware Mode
Connect your Arduino/sensor via USB.
```bash
python3 src/live_inference.py --port COM3 --model models/svm_lda_pipeline.joblib
# Note: Replace 'COM3' with your actual port (e.g., /dev/tty.usbmodem...)
```

##  Project Structure

- **`src/`**: Source code for live inference.
  - `live_inference.py`: Main script. Handles serial connection, data buffering, and predicting.
  - `preprocessing.py`: Implements signal filtering (Bandpass/Notch/Highpass) and Feature Extraction (DWT + Statistics).
  - `inference.py`: Wrapper class to load and use the trained `.joblib` models.
  - `data_collector.py`: Legacy script for simple data collection.
- **`models/`**: Saved Scikit-learn models (Pipeline dictionaries).
- **`notebooks/`**: Jupyter notebooks used for training and analysis.
- **`firmware/`**: Arduino code (`.ino`) for the EMG sensor.

##  Model Details
The system supports models saved as dictionaries containing a full pipeline:
- **Scaler**: StandardScaler
- **Dimensionality Reduction**: LDA
- **Classifier**: SVM or Naive Bayes
- **Input Features**: 1120 (32 Channels × 5 Wavelet Bands × 7 Statistical Features)
