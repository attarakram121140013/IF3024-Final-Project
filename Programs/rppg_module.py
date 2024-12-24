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
    if len(data) <= max(len(a), len(b)):
        return np.array(data)  # Return raw data if too short for filtering
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

def process_rppg_from_webcam(video):
    """
    Processes rPPG signals in real-time from a webcam feed.
    """
    cap = video
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

        # Real-time visualization of signal intensity
        if len(g_signals) > 33:  # Ensure enough data for filtering
            graph_height, graph_width = 100, 300
            graph_top_left = (10, frame.shape[0] - graph_height - 10)
            overlay = frame.copy()

            # Apply band-pass filter
            filtered_signal = band_pass_filter(g_signals, lowcut=0.8, highcut=3.0, fs=30)

            # Normalize filtered signal to scale it properly for visualization
            min_signal = np.min(filtered_signal)
            max_signal = np.max(filtered_signal)
            range_signal = max_signal - min_signal

            if range_signal > 0:
                # Normalize the signal to the range [0, graph_height]
                scaled_signal = (filtered_signal - min_signal) / range_signal * graph_height
            else:
                scaled_signal = np.zeros_like(filtered_signal)

            # Only visualize the last 300 frames
            if len(scaled_signal) > 300:
                scaled_signal = scaled_signal[-300:]

            for i in range(1, len(scaled_signal)):
                start_point = (graph_top_left[0] + i - 1, graph_top_left[1] + graph_height - int(scaled_signal[i - 1]))
                end_point = (graph_top_left[0] + i, graph_top_left[1] + graph_height - int(scaled_signal[i]))
                cv2.line(overlay, start_point, end_point, (0, 255, 0), 1)

            # Draw visualization box
            cv2.rectangle(frame, graph_top_left, (graph_top_left[0] + graph_width, graph_top_left[1] + graph_height), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Add text for live heart rate calculation
            peaks, _ = find_peaks(filtered_signal, prominence=0.5)
            heart_rate = 60 * len(peaks) / (len(filtered_signal) / 30) if len(filtered_signal) > 0 else 0
            cv2.putText(frame, f"Heart Rate: {heart_rate:.2f} bpm", (15, graph_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('rPPG Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plotting_rppg_signals(r_signals, g_signals, b_signals)

def plotting_rppg_signals(r_signals, g_signals, b_signals):
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

    # Display heart rate (bpm) in the title
    plt.title(f'rPPG Signal - Heart Rate: {heart_rate:.2f} bpm')

    # X-axis represents frames, Y-axis represents signal intensity
    plt.xlabel('Frame')
    plt.ylabel('Signal Intensity')

    # Add legend to explain the plot
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    process_rppg_from_webcam(cv2.VideoCapture(0))
