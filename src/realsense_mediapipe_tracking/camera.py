import os
import time
import pyrealsense2 as rs
import cv2
import numpy as np

class realsenseCamera:
    def __init__(self, width=640, height=480, fps=30):

        self.width = width
        self.height = height
        self.fps = fps

        self.recorder = False


        try:
            self.pipe = rs.pipeline()
            self.config = rs.config()

            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            self.profile = self.pipe.start(self.config)


            self.align = rs.align(rs.stream.color)

            print("Camera connected")
        except Exception as e:
            print("Likely no camera connected. \n", e)
    
    def get_frames(self):

        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def stream(self, display_rgb=True, display_depth=False, save=False):

            """Continuously stream and optionally display color+depth."""
            if save == True:
                self.start_recorder(output_path="output/")
            try:
                while True:
                    color_image, depth_image = self.get_frames()
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    if color_image is None or depth_image is None:
                        continue

                    if display_rgb:
                        cv2.imshow('Color', color_image)
                    if display_depth:
                        cv2.imshow('Depth', depth_colormap)
                    
                    if self.recorder == True:
                        self.color_writer.write(color_image)
                        self.depth_writer.write(depth_colormap)
                        


                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
            except Exception as e:
                print(f"Error streaming: {e}")
            finally:
                self.stop()
                
    def start_recorder(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.color_writer = cv2.VideoWriter(f"{output_path}rgb_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.depth_writer = cv2.VideoWriter(f"{output_path}depth_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height), isColor=True)

        self.recorder = True

    def stop(self):
        """Stop the streaming process."""
        self.pipe.stop()
        cv2.destroyAllWindows()
        print("camera stopped")

        if self.recorder:
            self.color_writer.release()
            self.depth_writer.release()

if __name__ == "__main__":
    cam = realsenseCamera()
    try:
        cam.stream(display_rgb=True, display_depth=True, save=False)
    finally:
        pass
        # cam.stop()
    
