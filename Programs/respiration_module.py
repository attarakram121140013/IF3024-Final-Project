"""
Digital Signal Processing - Final Project
Attar Akram Abdillah (121140013)
Natasya Ate Malem Bangun (121140052)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, medfilt
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import requests
import matplotlib.pyplot as plt

# Filters

def low_pass_filter(data, cutoff, fs, order=5):
    """
    Applies a low-pass Butterworth filter to the input signal.

    Args:
        data (np.ndarray): Input signal to be filtered.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter (default is 5).

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency (half of the sampling frequency)
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Design the Butterworth filter
    return filtfilt(b, a, data)  # Apply the filter to the input data

def median_filter(data, kernel_size=5):
    """
    Applies a median filter to the input signal.

    Args:
        data (np.ndarray): Input signal to be filtered.
        kernel_size (int): Size of the median filter window (default is 5).

    Returns:
        np.ndarray: Filtered signal.
    """
    return medfilt(data, kernel_size)  # Apply median filter to smooth the signal

def download_pose_model():
    """
    Downloads the MediaPipe Pose Landmarker model if not already available.
    Returns the file path of the model.

    Returns:
        str: File path to the downloaded pose landmarker model.
    """
    model_dir = "models"  # Directory to store the model
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
    model_path = os.path.join(model_dir, "pose_landmarker.task")  # Full path for the model

    # Download the model if it's not already present
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        response = requests.get(url, stream=True)  # Get the model from the URL
        with open(model_path, "wb") as f:
            f.write(response.content)  # Write the model data to the file
    
    return model_path  # Return the path to the model

def initialize_pose_landmarker():
    """
    Initializes and returns the MediaPipe Pose Landmarker.
    
    Returns:
        PoseLandmarker: The MediaPipe PoseLandmarker object.
    """
    # MediaPipe PoseLandmarker setup
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = download_pose_model()  # Get the model path

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path  # Set the model path
        ),
        running_mode=VisionRunningMode.IMAGE,  # Set the running mode for image processing
        num_poses=1,  # Process one pose at a time
        min_pose_detection_confidence=0.5,  # Minimum confidence for pose detection
        min_pose_presence_confidence=0.5,  # Minimum confidence for presence of a pose
        min_tracking_confidence=0.5  # Minimum confidence for pose tracking
    )

    # Create and return the PoseLandmarker object
    return PoseLandmarker.create_from_options(options)

def get_respiration_roi(image, pose_landmarker, scale_factor=0.7):
    """
    Detects the shoulders and returns the region of interest (ROI) for respiration signal extraction.
    
    Args:
        image (np.ndarray): Input video frame (image).
        pose_landmarker: MediaPipe pose detector object.
        scale_factor (float): Scaling factor for ROI size (default is 0.7).
    
    Returns:
        tuple: Coordinates of the ROI (left_x, top_y, right_x, bottom_y, center).
    """
    # Convert the frame to RGB for pose detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect the pose in the frame
    detection_result = pose_landmarker.detect(mp_image)
    
    # Raise an exception if no pose is detected
    if not detection_result.pose_landmarks:
        raise ValueError("No pose detected in the frame!")

    landmarks = detection_result.pose_landmarks[0]
    height, width = image.shape[:2]  # Get image dimensions

    # Get the coordinates of the shoulders (left and right)
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Calculate the center point between the shoulders
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

    # Calculate the adaptive size of the ROI based on frame dimensions
    x_size = int(abs(left_shoulder.x - right_shoulder.x) * width * scale_factor)
    y_size = x_size

    # Ensure the ROI is within image boundaries
    left_x = max(0, center_x - x_size)
    right_x = min(width, center_x + x_size)
    top_y = max(0, center_y - y_size)
    bottom_y = min(height, center_y + y_size)

    return (left_x, top_y, right_x, bottom_y, (center_x, center_y))  # Return ROI coordinates

def process_respiration_from_webcam(video):
    """
    Processes respiration signal in real-time from a webcam feed.
    
    Args:
        video: Video capture object for the webcam feed.
    """
    cap = video  # Capture object
    pose_landmarker = initialize_pose_landmarker()  # Initialize the pose detector
    respiration_signal = []  # List to store the respiration signal

    # Loop to read frames from the webcam
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # Exit loop if no frame is read

        try:
            # Get the region of interest (ROI) for respiration signal extraction
            left_x, top_y, right_x, bottom_y, center = get_respiration_roi(frame, pose_landmarker)
            roi = frame[top_y:bottom_y, left_x:right_x]  # Extract the ROI
            avg_intensity = np.mean(roi[:, :, 1])  # Get the average intensity from the green channel
            respiration_signal.append(avg_intensity)  # Append the intensity to the signal list

            # Filter the signal once we have enough data
            if len(respiration_signal) > 30:
                filtered_signal = median_filter(respiration_signal, kernel_size=5)
                filtered_signal = low_pass_filter(filtered_signal, cutoff=0.8, fs=30)

                # Visualize the filtered signal in real-time
                if len(filtered_signal) > 33:
                    graph_height, graph_width = 100, 300
                    graph_top_left = (10, frame.shape[0] - graph_height - 350)
                    overlay = frame.copy()

                    # Normalize the signal for visualization
                    min_signal = np.min(filtered_signal)
                    max_signal = np.max(filtered_signal)
                    range_signal = max_signal - min_signal

                    if range_signal > 0:
                        # Scale the signal to fit the graph height
                        scaled_signal = (filtered_signal - min_signal) / range_signal * graph_height
                    else:
                        scaled_signal = np.zeros_like(filtered_signal)

                    if len(scaled_signal) > 300:
                        scaled_signal = scaled_signal[-300:]  # Limit the length of the signal for display

                    # Draw the graph on the frame
                    for i in range(1, min(len(scaled_signal), graph_width)):
                        start_point = (graph_top_left[0] + i - 1, graph_top_left[1] + graph_height - int(scaled_signal[i - 1]))
                        end_point = (graph_top_left[0] + i, graph_top_left[1] + graph_height - int(scaled_signal[i]))
                        cv2.line(overlay, start_point, end_point, (0, 255, 0), 1)

                    # Draw the graph box and overlay on the frame
                    cv2.rectangle(frame, graph_top_left, (graph_top_left[0] + graph_width, graph_top_left[1] + graph_height), (255, 255, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                    # Add text to show the current respiration intensity
                    cv2.putText(frame, f"Respiration Intensity: {filtered_signal[-1]:.2f}", 
                                (15, graph_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw the bounding box and center point for the shoulders
            cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
        except ValueError:
            continue  # Skip if no pose is detected

        # Display the frame
        cv2.imshow('Respiration Tracking', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Plot the final respiration signal
    plotting_respiration_signals(respiration_signal)

def plotting_respiration_signals(respiration_signal):
    """
    Plots the respiration signal over time.

    Args:
        respiration_signal (list): The list containing the respiration signal data.
    """
    # Plot the filtered respiration signal
    if len(respiration_signal) > 0:
        filtered_signal = median_filter(respiration_signal, kernel_size=5)
        filtered_signal = low_pass_filter(filtered_signal, cutoff=0.8, fs=30)

        plt.figure(figsize=(10, 5))
        plt.plot(filtered_signal, label="Respiration Signal", color="green")
        plt.title("Respiration Signal Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Start processing from webcam
    process_respiration_from_webcam(cv2.VideoCapture(0))
