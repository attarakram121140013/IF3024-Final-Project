import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, medfilt
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import requests
import matplotlib.pyplot as plt

#Filters
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

def median_filter(data, kernel_size=5):
    """
    Applies a median filter to the input signal.

    Args:
        data (np.ndarray): Input signal.
        kernel_size (int): Size of the median filter window.

    Returns:
        np.ndarray: Filtered signal.
    """
    return medfilt(data, kernel_size)

def download_pose_model():
    """
    Downloads the MediaPipe Pose Landmarker model if not already available.
    Returns the file path of the model.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker.task")
    
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
    
    return model_path

def initialize_pose_landmarker():
    """
    Initializes and returns the MediaPipe Pose Landmarker.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = download_pose_model()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return PoseLandmarker.create_from_options(options)

def get_respiration_roi(image, pose_landmarker, scale_factor=0.7):
    """
    Detects the shoulders and returns the ROI for respiration signal extraction.

    Args:
        image (np.ndarray): Input video frame.
        pose_landmarker: MediaPipe pose detector.
        scale_factor (float): Scaling factor for ROI size.

    Returns:
        tuple: Coordinates of the ROI (left_x, top_y, right_x, bottom_y, center).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = pose_landmarker.detect(mp_image)
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in the frame!")

    landmarks = detection_result.pose_landmarks[0]
    height, width = image.shape[:2]

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

    # Adaptive ROI size based on frame dimensions and scaling factor
    x_size = int(abs(left_shoulder.x - right_shoulder.x) * width * scale_factor)
    y_size = x_size

    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)

    return (left_x, top_y, right_x, bottom_y, (center_x, center_y))

def process_respiration_from_webcam(video):
    """
    Processes respiration signal in real-time from a webcam feed.
    """
    cap = video
    pose_landmarker = initialize_pose_landmarker()
    respiration_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            left_x, top_y, right_x, bottom_y, center = get_respiration_roi(frame, pose_landmarker)
            roi = frame[top_y:bottom_y, left_x:right_x]
            avg_intensity = np.mean(roi[:, :, 1])  # Green channel
            respiration_signal.append(avg_intensity)

            if len(respiration_signal) > 30:
                filtered_signal = median_filter(respiration_signal, kernel_size=5)
                filtered_signal = low_pass_filter(filtered_signal, cutoff=0.5, fs=30)

                # Real-time visualization of signal intensity
                if len(filtered_signal) > 33:  # Ensure enough data for filtering
                    graph_height, graph_width = 100, 300
                    graph_top_left = (10, frame.shape[0] - graph_height - 350)
                    overlay = frame.copy()

                    # Normalize filtered signal to scale it properly for visualization
                    min_signal = np.min(filtered_signal)
                    max_signal = np.max(filtered_signal)
                    range_signal = max_signal - min_signal

                    if range_signal > 0:
                        # Normalize the signal to the range [0, graph_height]
                        scaled_signal = (filtered_signal - min_signal) / range_signal * graph_height
                    else:
                        scaled_signal = np.zeros_like(filtered_signal)

                    if len(scaled_signal) > 300:
                        scaled_signal = scaled_signal[-300:]

                    for i in range(1, min(len(scaled_signal), graph_width)):
                        start_point = (graph_top_left[0] + i - 1, graph_top_left[1] + graph_height - int(scaled_signal[i - 1]))
                        end_point = (graph_top_left[0] + i, graph_top_left[1] + graph_height - int(scaled_signal[i]))
                        cv2.line(overlay, start_point, end_point, (0, 255, 0), 1)

                    # Draw visualization box
                    cv2.rectangle(frame, graph_top_left, (graph_top_left[0] + graph_width, graph_top_left[1] + graph_height), (255, 255, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                    # Add text for live respiration signal reading
                    cv2.putText(frame, f"Respiration Intensity: {filtered_signal[-1]:.2f}", 
                                (15, graph_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
        except ValueError:
            continue

        cv2.imshow('Respiration Tracking', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plotting_respiration_signals(respiration_signal)

def plotting_respiration_signals(respiration_signal):
    # Plot the final signal
    if len(respiration_signal) > 0:
        filtered_signal = median_filter(respiration_signal, kernel_size=5)
        filtered_signal = low_pass_filter(filtered_signal, cutoff=0.5, fs=30)

        plt.figure(figsize=(10, 5))
        plt.plot(filtered_signal, label="Respiration Signal", color="green")
        plt.title("Respiration Signal Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    process_respiration_from_webcam(cv2.VideoCapture(0))
