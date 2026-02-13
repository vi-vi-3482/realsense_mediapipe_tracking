import copy
import os
import time
import mediapipe
import cv2
import numpy as np
import pyrealsense2 as rs
from . import camera

class handTrack:
    """
    Class to track hands using Mediapipe and RealSense camera. This class uses the RealSense camera to capture frames and then uses 
    Mediapipe to detect hands in those frames. 
    Requires a camera object from camera.realsenseCamera to be passed in as a parameter.
    """
    def __init__(self, cam):
        self.cam = cam
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       
            max_num_hands=2,               
            model_complexity=1,            
            min_detection_confidence=0.5,  
            min_tracking_confidence=0.5    
        )
        self.mp_draw = mediapipe.solutions.drawing_utils
        self.mp_drawing_styles = mediapipe.solutions.drawing_styles

        self.recorder = False 

    def stream(self, marks_to_show=[0, 4, 8, 12, 16, 20], save=False):
        """Continuously stream and optionally display color frame with hand landmarks and bones.
        Prints the XYZ positions (meters) of landmarks specified in marks_to_show.
        Save the frames to output/ directory if save=True.
        """
        if save == True:
            self.start_recorder(output_path="output/")
        while True:
            color_image, depth_image, depth_frame = self.cam.get_frames()
            h, w, _ = color_image.shape

            landmarks_xyz, results = self.tracking(color_image, depth_frame)

            if not landmarks_xyz:
                cv2.imshow("Landmark Tracking", color_image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            for hand_xyz, hand_results in zip(landmarks_xyz, results.multi_hand_landmarks):
                for idx in marks_to_show:
                    X, Y, Z = hand_xyz[idx]
                    lm = hand_results.landmark[idx]
                    px, py = int(lm.x * w), int(lm.y * h)

                    cv2.circle(color_image, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{idx}: {X:.2f},{Y:.2f},{Z:.2f}m",
                                (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 255), 1)

                self.mp_draw.draw_landmarks(
                    color_image,
                    hand_results,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            cv2.imshow("Landmark Tracking", color_image)

            if self.recorder == True:
                self.hand_writer.write(color_image)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    def tracking(self, color_image, depth_frame):
        """Returns a list of (x, y, z) coordinates for each hand landmark in meters relative to centre of the camera.
        To use the, call in a while loop, inputing the realsense color image and depth frames then save the list in the desired format."""
        h, w, _ = color_image.shape
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return [], results

        hand_landmarks_xyz = []

        for hand_landmarks in results.multi_hand_landmarks:
            xyz_points = []
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Clamp to image bounds
                cx = np.clip(cx, 0, w - 1)
                cy = np.clip(cy, 0, h - 1)

                depth = depth_frame.get_distance(cx, cy)
                if depth == 0:
                    xyz_points.append((np.nan, np.nan, np.nan))
                    continue

                x, y, z = rs.rs2_deproject_pixel_to_point(self.cam.depth_intrinsics_aligned, [cx, cy], depth)

                xyz_points.append((x, y, z))

            hand_landmarks_xyz.append(xyz_points)

        return hand_landmarks_xyz, results           

    def start_recorder(self, output_path):
        """Stars the cv2 video writer."""
        fps = self.cam.fps
        width, height = self.cam.width, self.cam.height
        os.makedirs(output_path, exist_ok=True)
        self.hand_writer = cv2.VideoWriter(f"{output_path}/hand_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        self.recorder = True

    def stop_recorder(self):
        """Releases the cv2 video writer."""
        if self.recorder == True:
            self.hand_writer.release()
            self.recorder = False

    def record(self, video_time):
        """Records a for a set duration (seconds)"""
        self.start_recorder(output_path="output/")

        start = time.monotonic()

        try:
            while time.monotonic() - start < video_time:

                frame_start = time.monotonic()
                
                color_image, depth_image = self.get_frames()
                if color_image is None or depth_image is None:
                    continue
                


                elapsed = time.monotonic() - frame_start
                wait = self.frame_time - elapsed
                if wait > 0:
                    time.sleep(wait)


        except Exception as e:
            print(e)
        finally:
            self.stop_recorder()
            self.stop()


if __name__ == "__main__":
    cam = camera.realsenseCamera()
    track = handTrack(cam)
    track.stream()
