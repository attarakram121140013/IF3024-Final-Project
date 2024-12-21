from respiration_module import process_respiration_from_webcam
from rppg_module import process_rppg_from_webcam
import threading

def main():
    """
    Runs both the respiration and rPPG processing in parallel using threading.
    """
    # Create threads for each signal processing function
    respiration_thread = threading.Thread(target=process_respiration_from_webcam)
    rppg_thread = threading.Thread(target=process_rppg_from_webcam)

    # Start threads
    respiration_thread.start()
    rppg_thread.start()

    # Wait for threads to complete
    respiration_thread.join()
    rppg_thread.join()

if __name__ == "__main__":
    main()
