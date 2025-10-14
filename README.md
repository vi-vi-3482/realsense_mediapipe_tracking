# Realsense Mediapipe Tracking

This is a project to integrate the use of a Realsense d435 depth camera with Mediapipe for tracking the position of a subjects hand. 

Using UV for package management.

Mediapipe requires python 3.12 or older unless built from source.

### Example Streaming
``` python
import time
from src.realsense_mediapipe_tracking.camera import realsenseCamera
from src.realsense_mediapipe_tracking.hand_tracking import handTrack

def main():
    # Initialize the RealSense camera with desired parameters
    cam = realsenseCamera(width=640, height=480, fps=30)

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

if __name__ == "__main__":
    main()
```

### Example direct x y z coordinate printing
```python
# Example usage for printing the XYZ positions of all hand landmarks

import camera
from src.realsense_mediapipe_tracking.hand_tracking import handTrack

def main():
    # Initialize the RealSense camera
    cam = camera.realsenseCamera()

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

    except Exception a e:
        print(f"An error occurred: {e}")

    finally:
        cam.stop()

if __name__ == "__main__":
    main()
```

Required external libraries:
https://github.com/IntelRealSense/librealsense

Licensed under GNU GPLv3