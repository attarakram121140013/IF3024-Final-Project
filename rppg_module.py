import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def low_pass_filter(data, cutoff, fs, order=5):
    """
    Applies a low-pass Butterworth filter to the input signal.

    Args:
        data (np.ndarray): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def band_pass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a band-pass Butterworth filter to the input signal.

    Args:
        data (np.ndarray): Input signal.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_rgb_signals(frame, bbox):
    """
    Extracts the average RGB signals from the bounding box region of the frame.

    Args:
        frame (np.ndarray): Input video frame.
        bbox (tuple): Bounding box coordinates (x, y, width, height).

    Returns:
        tuple: Average RGB values as (R, G, B).
    """
    x, y, width, height = bbox
    roi = frame[y:y + height, x:x + width]
    r_signal = np.mean(roi[:, :, 2])  # Red channel
    g_signal = np.mean(roi[:, :, 1])  # Green channel
    b_signal = np.mean(roi[:, :, 0])  # Blue channel
    return r_signal, g_signal, b_signal

def process_rppg_from_webcam():
    """
    Processes rPPG signals in real-time from a webcam feed.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    r_signals, g_signals, b_signals = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Extract RGB signals from the face bounding box
                r, g, b = extract_rgb_signals(frame, (x, y, width, height))
                r_signals.append(r)
                g_signals.append(g)
                b_signals.append(b)

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('rPPG Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Process signals for rPPG
    if len(r_signals) > 0:
        process_rppg_signals(r_signals, g_signals, b_signals)

def process_rppg_signals(r_signals, g_signals, b_signals):
    """
    Processes the extracted RGB signals to calculate and visualize the rPPG signal.

    Args:
        r_signals (list): Red channel signals.
        g_signals (list): Green channel signals.
        b_signals (list): Blue channel signals.
    """
    # Stack RGB signals
    rgb_signals = np.array([r_signals, g_signals, b_signals])

    # Band-pass filter for the green channel (dominant in rPPG)
    fs = 30  # Assuming 30 FPS
    lowcut = 0.8  # Lower bound for heart rate frequencies
    highcut = 3.0  # Upper bound for heart rate frequencies
    filtered_signal = band_pass_filter(rgb_signals[1], lowcut, highcut, fs)

    # Detect peaks for heart rate calculation
    peaks, _ = find_peaks(filtered_signal, prominence=0.5)
    heart_rate = 60 * len(peaks) / (len(filtered_signal) / fs)

    # Plot the rPPG signal and peaks
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_signal, label='Filtered rPPG Signal', color='green')
    plt.plot(peaks, filtered_signal[peaks], 'x', label='Peaks', color='red')
    plt.title(f'rPPG Signal - Heart Rate: {heart_rate:.2f} bpm')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    process_rppg_from_webcam()
