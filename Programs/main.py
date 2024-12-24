"""
Digital Signal Processing - Final Project
Attar Akram Abdillah (121140013)
Natasya Ate Malem Bangun (121140052)
"""

import cv2
import threading
import queue
import numpy as np
from respiration_module import get_respiration_roi, median_filter, low_pass_filter, initialize_pose_landmarker
from rppg_module import extract_rgb_signals, band_pass_filter
import mediapipe as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class SharedVideoProcessor:
    """
    A class that handles video capture, respiration signal extraction, and rPPG signal processing.

    Attributes:
        cap (cv2.VideoCapture): Video capture object to access the webcam.
        running (bool): Flag to control the video processing loop.
        frame_lock (threading.Lock): Lock to ensure thread-safe access to frames.
        current_frame (numpy.ndarray): The current frame captured from the video.
        pose_landmarker: Initialized pose landmarker for respiration signal extraction.
        mp_face_detection: MediaPipe face detection module.
        face_detection: Face detection object for detecting faces in frames.
        respiration_signal (list): List to store extracted respiration signals.
        r_signals, g_signals, b_signals (list): Lists to store RGB channel values for rPPG.
    """

    def __init__(self):
        """
        Initializes the video capture, pose landmarker, and face detection, 
        and sets up signal storage.
        """
        self.cap = cv2.VideoCapture(0)  # Initialize the video capture with the default webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the frame width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the frame height
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frames per second

        self.running = True  # Flag to indicate if video processing should continue
        self.frame_lock = threading.Lock()  # Lock for thread-safe access to frames
        self.current_frame = None  # Variable to store the current video frame

        # Initialize pose landmarker for respiration signal extraction
        self.pose_landmarker = initialize_pose_landmarker()
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Initialize lists to store the signals
        self.respiration_signal = []  # List to store respiration signals
        self.r_signals = []  # List to store red channel values for rPPG
        self.g_signals = []  # List to store green channel values for rPPG
        self.b_signals = []  # List to store blue channel values for rPPG

    def start(self):
        """
        Starts the video capture and processing threads.

        This method initializes the threads for capturing frames and processing signals.
        """
        self.capture_thread = threading.Thread(target=self.capture_frames)  # Thread for capturing frames
        self.processing_thread = threading.Thread(target=self.process_frames)  # Thread for processing frames
        
        self.capture_thread.start()  # Start the frame capture thread
        self.processing_thread.start()  # Start the frame processing thread

    def capture_frames(self):
        """
        Captures frames from the webcam in a separate thread.

        This method runs in the capture thread and continuously captures frames from the webcam.
        """
        while self.running:
            ret, frame = self.cap.read()  # Read a frame from the video capture
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()  # Store the current frame

    def process_frames(self):
        """
        Processes the captured frames to extract respiration and rPPG signals.

        This method runs in the processing thread and processes the captured frames to extract
        respiration and rPPG signals. It also displays the processed frames with the signals.
        """
        while self.running:
            with self.frame_lock:
                if self.current_frame is None:  # Skip if no frame has been captured yet
                    continue
                frame = self.current_frame.copy()  # Copy the current frame for processing

            # Process respiration signal
            try:
                left_x, top_y, right_x, bottom_y, center = get_respiration_roi(frame, self.pose_landmarker)
                roi = frame[top_y:bottom_y, left_x:right_x]  # Extract Region of Interest (ROI)
                avg_intensity = np.mean(roi[:, :, 1])  # Calculate average intensity from the green channel
                self.respiration_signal.append(avg_intensity)  # Store the respiration signal

                # Draw respiration ROI on the frame
                cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
            except ValueError:
                pass  # If respiration ROI cannot be detected, pass

            # Process rPPG signal
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            results = self.face_detection.process(frame_rgb)  # Detect faces in the frame

            if results.detections:  # If any faces are detected
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)

                    # Extract RGB signals from the detected face region
                    r, g, b = extract_rgb_signals(frame, (x, y, width, height))
                    self.r_signals.append(r)  # Store red channel signal
                    self.g_signals.append(g)  # Store green channel signal
                    self.b_signals.append(b)  # Store blue channel signal

                    # Draw bounding box around the detected face
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Visualize the respiration signal if enough data has been collected
            if len(self.respiration_signal) > 30:
                self.visualize_respiration(frame)

            # Visualize the rPPG signal if enough data has been collected
            if len(self.g_signals) > 33:
                self.visualize_rppg(frame)

            # Display the frame with annotations
            cv2.imshow('Respiration and rPPG Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
                self.running = False

    def visualize_respiration(self, frame):
        """
        Visualizes the respiration signal by filtering and plotting it on the frame.

        Args:
            frame (numpy.ndarray): The current frame from the video capture.

        """
        # Filter and process the respiration signal
        filtered_signal = median_filter(self.respiration_signal, kernel_size=5)
        filtered_signal = low_pass_filter(filtered_signal, cutoff=0.5, fs=30)

        # Visualization settings for the respiration graph
        graph_height, graph_width = 100, 300
        graph_top_left = (10, frame.shape[0] - graph_height - 350)
        
        if len(filtered_signal) > 33:
            self.draw_signal(frame, filtered_signal, graph_top_left, graph_height, graph_width, 
                           "Respiration Intensity", (0, 255, 0))

    def visualize_rppg(self, frame):
        """
        Visualizes the rPPG signal by filtering it and plotting heart rate information on the frame.

        Args:
            frame (numpy.ndarray): The current frame from the video capture.
        """
        # Filter the rPPG signal
        filtered_signal = band_pass_filter(self.g_signals, lowcut=0.8, highcut=3.0, fs=30)
        
        # Calculate heart rate from the rPPG signal
        peaks, _ = find_peaks(filtered_signal, prominence=0.5)
        heart_rate = 60 * len(peaks) / (len(filtered_signal) / 30) if len(filtered_signal) > 0 else 0

        # Visualization settings for the rPPG graph
        graph_height, graph_width = 100, 300
        graph_top_left = (10, frame.shape[0] - graph_height - 10)
        
        self.draw_signal(frame, filtered_signal, graph_top_left, graph_height, graph_width,
                        f"Heart Rate: {heart_rate:.2f} bpm", (0, 255, 0))

    def draw_signal(self, frame, signal, graph_top_left, graph_height, graph_width, label, color):
        """
        Draws the signal graph on the frame.

        Args:
            frame (numpy.ndarray): The current frame from the video capture.
            signal (list): The signal to be drawn.
            graph_top_left (tuple): Top-left coordinates for the graph.
            graph_height (int): Height of the graph.
            graph_width (int): Width of the graph.
            label (str): Label to display above the graph.
            color (tuple): Color for the signal line.
        """
        overlay = frame.copy()  # Copy of the frame to draw the signal overlay
        
        # Normalize the signal to fit the graph's height
        if len(signal) > 0:
            min_signal = np.min(signal)
            max_signal = np.max(signal)
            range_signal = max_signal - min_signal
            
            if range_signal > 0:
                scaled_signal = (signal - min_signal) / range_signal * graph_height
            else:
                scaled_signal = np.zeros_like(signal)

            if len(scaled_signal) > graph_width:
                scaled_signal = scaled_signal[-graph_width:]

            # Draw the signal line on the graph
            for i in range(1, len(scaled_signal)):
                start_point = (graph_top_left[0] + i - 1, 
                             graph_top_left[1] + graph_height - int(scaled_signal[i - 1]))
                end_point = (graph_top_left[0] + i, 
                           graph_top_left[1] + graph_height - int(scaled_signal[i]))
                cv2.line(overlay, start_point, end_point, color, 1)

        # Draw the graph box and label text
        cv2.rectangle(frame, graph_top_left, 
                     (graph_top_left[0] + graph_width, graph_top_left[1] + graph_height), 
                     (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, label, (15, graph_top_left[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def stop(self):
        """
        Stops the video processing and captures threads, releases the video capture, 
        and closes any OpenCV windows.
        """
        self.running = False
        self.capture_thread.join()  # Wait for the capture thread to finish
        self.processing_thread.join()  # Wait for the processing thread to finish
        self.cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows
        
        # Plot final signals after processing is complete
        self.plot_final_signals()

    def plot_final_signals(self):
        """
        Plots the final respiration and rPPG signals after video processing is stopped.
        """
        # Plot the respiration signal
        if len(self.respiration_signal) > 0:
            plt.figure(figsize=(10, 5))
            filtered_signal = median_filter(self.respiration_signal, kernel_size=5)
            filtered_signal = low_pass_filter(filtered_signal, cutoff=0.5, fs=30)
            plt.plot(filtered_signal, label="Respiration Signal", color="green")
            plt.title("Respiration Signal Over Time")
            plt.xlabel("Frame")
            plt.ylabel("Intensity")
            plt.legend()
            plt.grid()
            plt.show()

        # Plot the rPPG signal
        if len(self.g_signals) > 0:
            filtered_signal = band_pass_filter(self.g_signals, lowcut=0.8, highcut=3.0, fs=30)
            peaks, _ = find_peaks(filtered_signal, prominence=0.5)
            heart_rate = 60 * len(peaks) / (len(filtered_signal) / 30)

            plt.figure(figsize=(10, 6))
            plt.plot(filtered_signal, label='Filtered rPPG Signal', color='green')
            plt.plot(peaks, filtered_signal[peaks], 'x', label='Peaks', color='red')
            plt.title(f'rPPG Signal - Heart Rate: {heart_rate:.2f} bpm')
            plt.xlabel('Frame')
            plt.ylabel('Signal Intensity')
            plt.legend()
            plt.show()

def main():
    """
    The main function that runs the video processing system.
    """
    processor = SharedVideoProcessor()  # Initialize the SharedVideoProcessor
    try:
        processor.start()  # Start the video capture and processing threads
        while processor.running:
            pass
    except KeyboardInterrupt:
        print("\nStopping the program...")  # Handle interruption
    finally:
        processor.stop()  # Ensure the processor is stopped when the program ends

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
