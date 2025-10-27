from camera import realsenseCamera
from hand_tracking import handTrack
import time
import csv

def streaming(vid_width, vid_height, fps):
    """A basic streaming function showing the color stream with hand tracking landmarks and coordinates."""
    # Initialize the RealSense camera with desired parameters
    cam = realsenseCamera(width=vid_width, height=vid_height, fps=fps)

    # Initialize the hand tracking class with the camera object
    track = handTrack(cam)

    try:
        # Start streaming
        print("Starting stream...")
        track.stream()
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop the camera and release resources
        cam.stop()

def print_landmarks():
    """A basic function to print out the xyz coordinates of each hand landmark when measured"""
    # Initialize the RealSense camera
    cam = realsenseCamera()

    # Initialize the hand tracking class with the camera object
    tracker = handTrack(cam)

    try:
        while True:
            color_image, depth_image, depth_frame = tracker.cam.get_frames()
            
            if color_image is None or depth_image is None:
                continue

            landmarks_xyz, results = tracker.tracking(color_image, depth_frame)

            # Print the XYZ coordinates for each hand
            for hand_xyz in landmarks_xyz:
                for idx, (X, Y, Z) in enumerate(hand_xyz):
                    print(f"Landmark {idx}: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cam.stop()

def save_to_csv(save_path, interval=1):
    cam = realsenseCamera()
    tracker = handTrack(cam)

    header = ['time', ]
    with open(save_path, 'w', newline='') as file:


