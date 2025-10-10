import os
import time
import mediapipe as mp
import cv2
import realsense_mediapipe_tracking as rs

class handTrack:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def stream(self):
        pass

     # Prints the coordinates of the landmark in meters
        cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(image,
                    f"{idx}: {X:.2f},{Y:.2f},{Z:.2f}m",
                    (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1)

        # Draw the hand skeleton
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    def tracking(self, rgb_image, depth_frame):
        """
        Returns a list of 3D coordinates (x, y, z) for each hand landmark in camera space (meters).
        """
        h, w, _ = rgb_image.shape
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        hand_points_3d = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_3d = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)

                    # Bounds check
                    if px < 0 or py < 0 or px >= w or py >= h:
                        landmarks_3d.append(None)
                        continue

                    depth = depth_frame.get_distance(px, py)
                    if depth == 0:
                        landmarks_3d.append(None)
                        continue

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], depth)
                    landmarks_3d.append((X, Y, Z))
                hand_points_3d.append(landmarks_3d)

        return hand_points_3d




                   


    def start_recorder(self, output_path, fps, width, height):
        os.makedirs(output_path, exist_ok=True)
        self.hand_writer = cv2.VideoWriter(f"{output_path}rgb_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    def stop_recorder(self):
        if self.recorder == True:
            pass
            self.recorder = False

    def record(self, video_time):
        self.start_recorder(output_path="output/")

        start = time.monotonic()

        try:
            while time.monotonic() - start < video_time:

                frame_start = time.monotonic()
                
                color_image, depth_image = self.get_frames()
                if color_image is None or depth_image is None:
                    continue
                self.color_writer.write(color_image)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                self.depth_writer.write(depth_colormap)

                elapsed = time.monotonic() - frame_start
                wait = self.frame_time - elapsed
                if wait > 0:
                    time.sleep(wait)


        except Exception as e:
            print(e)
        finally:
            self.stop_recorder()
            self.stop()
