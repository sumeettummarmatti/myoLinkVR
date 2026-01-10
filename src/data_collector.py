import serial
import time
import csv
import os

# --- CONFIGURATION ---
COM_PORT = '/dev/tty.HC-05'       # <--- CHANGE THIS to your HC-05 COM Port
BAUD_RATE = 9600
OUTPUT_FOLDER = '../data/raw'

# Ensure output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Create a unique filename based on time
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"{OUTPUT_FOLDER}/session_{timestamp}.csv"

print(f"Connecting to {COM_PORT}...")

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for connection to stabilize
    print("Connected!")
except Exception as e:
    print(f"Error connecting to Bluetooth: {e}")
    exit()

print(f"Recording data to {filename}")
print("Press Ctrl+C to stop recording.")

# Open the file and write headers
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Raw_EMG"])  # CSV Header

    try:
        start_time = time.time()
        while True:
            if ser.in_waiting > 0:
                try:
                    # Read line from Arduino
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Check if it matches your Arduino's format "Emg signals: 123"
                    if "Emg signals:" in line:
                        value_str = line.split(":")[1].strip()
                        
                        if value_str.isdigit():
                            current_time = time.time() - start_time
                            raw_val = int(value_str)
                            
                            # Write to CSV
                            writer.writerow([f"{current_time:.4f}", raw_val])
                            
                            # Optional: Print to screen so you know it's working
                            print(f"Recorded: {raw_val}")
                            
                except ValueError:
                    pass  # Ignore bad data packets

    except KeyboardInterrupt:
        print(f"\nRecording stopped. Data saved to {filename}")
        ser.close()