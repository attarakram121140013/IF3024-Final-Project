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
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.frame_lock = threading.Lock()
        self.current_frame = None

        # Initialize pose landmarker
        self.pose_landmarker = initialize_pose_landmarker()
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Signal storage
        self.respiration_signal = []
        self.r_signals = []
        self.g_signals = []
        self.b_signals = []

    def start(self):
        # Start the capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread = threading.Thread(target=self.process_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()

    def process_frames(self):
        while self.running:
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

            # Process respiration
            try:
                left_x, top_y, right_x, bottom_y, center = get_respiration_roi(frame, self.pose_landmarker)
                roi = frame[top_y:bottom_y, left_x:right_x]
                avg_intensity = np.mean(roi[:, :, 1])  # Green channel
                self.respiration_signal.append(avg_intensity)

                # Draw respiration ROI
                cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
            except ValueError:
                pass

            # Process rPPG
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)

                    # Extract RGB signals
                    r, g, b = extract_rgb_signals(frame, (x, y, width, height))
                    self.r_signals.append(r)
                    self.g_signals.append(g)
                    self.b_signals.append(b)

                    # Draw face detection box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Display signals if enough data
            if len(self.respiration_signal) > 30:
                self.visualize_respiration(frame)

            if len(self.g_signals) > 33:
                self.visualize_rppg(frame)

            # Show the frame
            cv2.imshow('Respiration and rPPG Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def visualize_respiration(self, frame):
        filtered_signal = median_filter(self.respiration_signal, kernel_size=5)
        filtered_signal = low_pass_filter(filtered_signal, cutoff=0.5, fs=30)

        # Visualization code for respiration
        graph_height, graph_width = 100, 300
        graph_top_left = (10, frame.shape[0] - graph_height - 350)
        
        if len(filtered_signal) > 33:
            self.draw_signal(frame, filtered_signal, graph_top_left, graph_height, graph_width, 
                           "Respiration Intensity", (0, 255, 0))

    def visualize_rppg(self, frame):
        filtered_signal = band_pass_filter(self.g_signals, lowcut=0.8, highcut=3.0, fs=30)
        
        # Calculate heart rate
        peaks, _ = find_peaks(filtered_signal, prominence=0.5)
        heart_rate = 60 * len(peaks) / (len(filtered_signal) / 30) if len(filtered_signal) > 0 else 0

        # Visualization code for rPPG
        graph_height, graph_width = 100, 300
        graph_top_left = (10, frame.shape[0] - graph_height - 10)
        
        self.draw_signal(frame, filtered_signal, graph_top_left, graph_height, graph_width,
                        f"Heart Rate: {heart_rate:.2f} bpm", (0, 255, 0))

    def draw_signal(self, frame, signal, graph_top_left, graph_height, graph_width, label, color):
        overlay = frame.copy()
        
        # Normalize signal
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

            # Draw signal
            for i in range(1, len(scaled_signal)):
                start_point = (graph_top_left[0] + i - 1, 
                             graph_top_left[1] + graph_height - int(scaled_signal[i - 1]))
                end_point = (graph_top_left[0] + i, 
                           graph_top_left[1] + graph_height - int(scaled_signal[i]))
                cv2.line(overlay, start_point, end_point, color, 1)

        # Draw visualization box and text
        cv2.rectangle(frame, graph_top_left, 
                     (graph_top_left[0] + graph_width, graph_top_left[1] + graph_height), 
                     (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, label, (15, graph_top_left[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Plot final signals
        self.plot_final_signals()

    def plot_final_signals(self):
        # Plot respiration signal
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

        # Plot rPPG signal
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
    processor = SharedVideoProcessor()
    try:
        processor.start()
        while processor.running:
            pass
    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()