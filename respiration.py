import cv2
import dlib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define a function to calculate respiratory rate
def calculate_respiratory_rate(motion_signal, fps=30):
    # Apply a bandpass filter to the motion signal
    b, a = butter(3, [0.05, 0.5], btype='bandpass', fs=fps)
    filtered_signal = filtfilt(b, a, motion_signal)
    
    # Compute the power spectrum of the filtered signal
    spectrum = np.abs(fft(filtered_signal))**2
    
    # Find the dominant frequency in the power spectrum
    freqs = np.fft.fftfreq(len(spectrum), 1/fps)
    idx = np.argmax(spectrum[1:int(len(spectrum)/2)])
    respiratory_rate = freqs[idx+1]*60
    
    return respiratory_rate

# Start video capture
cap = cv2.VideoCapture(0)

# Define variables for calculating respiratory rate
prev_frame_roi = None
prev_frame_gray = None
fps = cap.get(cv2.CAP_PROP_FPS)

while True:

    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray, 1)

    # Loop through each face
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Convert the landmarks to a numpy array
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract the region of interest (ROI) around the upper face
        roi = landmarks_array[27:36]

        # Calculate the bounding box of the ROI
        x, y, w, h = cv2.boundingRect(roi)
        
        # Extract the ROI from the grayscale image
        frame_roi_gray = gray[y:y+h, x:x+w]

        # Apply a Gaussian filter to the ROI
        frame_roi_gray = cv2.GaussianBlur(frame_roi_gray, (5, 5), 0)

        # If this is not the first frame, calculate the motion signal
        if prev_frame_roi is not None:
            # Calculate the optical flow between the previous and current frames
            flow = cv2.calcOpticalFlowFarneback(prev_frame_roi, frame_roi_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate the magnitude of the flow vector at each pixel in the ROI
            flow_mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

            # Calculate the respiratory rate using the motion signal
            rr = calculate_respiratory_rate(flow_mag, fps)

            # Print the respiratory rate
            print(f"Respiratory rate: {round(rr, 1)} breaths per minute")

            # Draw a rectangle around the ROI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw the respiratory rate on the frame
            cv2.putText(frame, f"RR: {round(rr, 1)} bpm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Show the motion signal and power spectrum
            if prev_frame_roi is not None:
                
            

                
                 # Draw the motion signal
                motion_signal = flow_mag
                motion_signal *= (100/motion_signal.max())
                motion_signal = motion_signal.astype(int)
                motion_signal_img = np.zeros((100, motion_signal.size, 3), dtype=np.uint8)
                motion_signal_img[:, :, 1] = motion_signal
                motion_signal_img = np.flip(motion_signal_img, axis=0)
                cv2.imshow('Motion Signal', motion_signal_img)


                spectrum = np.abs(fft(motion_signal))**2
                freqs = np.fft.fftfreq(len(spectrum), 1/fps)
                idx = np.argmax(spectrum[1:int(len(spectrum)/2)])
                respiratory_rate = freqs[idx+1]*60
                peaks, _ = find_peaks(spectrum[1:int(len(spectrum)/2)], height=100)
                plt.plot(freqs[1:int(len(spectrum)/2)], spectrum[1:int(len(spectrum)/2)])
                plt.plot(freqs[peaks], spectrum[peaks], 'x')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.xlim([0, 2])
                plt.ylim([0, 100000])
                plt.title(f"Respiratory Rate: {round(respiratory_rate, 1)} bpm")
                plt.show()


                prev_frame_roi = frame_roi_gray
                prev_frame_gray = gray

# Display the frame
                cv2.imshow('Respiratory Rate Monitor', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cap.release()
                cv2.destroyAllWindows()
        
    





                


    # Compute and plot the power spectrum
                
   

# Save the current ROI and grayscale frame for the next iteration
   
# Exit the loop if the 'q' key is pressed

