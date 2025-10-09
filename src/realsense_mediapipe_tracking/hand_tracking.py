import os
import time
import mediapipe as mp
import cv2

class handTrack:
    def __init__(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

    def stream(self):
        pass

    def tracking(self, rgb_image, depth_image):
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)


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
