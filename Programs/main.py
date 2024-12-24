import cv2
import threading
import rppg_module
import respiration_module

# Global flag to indicate if the program should exit
exit_flag = False

def process_rppg(cap):
    """
    Processes rPPG from the webcam feed in a separate thread.
    """
    while not exit_flag:
        rppg_module.process_rppg_from_webcam(cap)

def process_respiration(cap):
    """
    Processes respiration from the webcam feed in a separate thread.
    """
    while not exit_flag:
        respiration_module.process_respiration_from_webcam(cap)

def process_shared_webcam():
    """
    Processes webcam feed for rPPG and respiration modules using threading.
    """
    global exit_flag

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    # Create threads for rPPG and respiration
    respiration_thread = threading.Thread(target=process_respiration, args=(cap,))
    rppg_thread = threading.Thread(target=process_rppg, args=(cap,))

    # Start the threads
    respiration_thread.start()
    rppg_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Exit condition when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Wait for both threads to finish
    respiration_thread.join()
    rppg_thread.join()

if __name__ == "__main__":
    process_shared_webcam()
