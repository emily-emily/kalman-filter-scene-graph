"""
A helper to output a video from a list of frames.
"""

import cv2
import numpy as np

def generate_video(filename, frames, fps=2):
    """
    Generates a video from a list of frames.

    Args:
    - filename (str): The output filename.
    - frames (list): List of PIL.Image frames.
    - fps (int): Frames per second for the video.
    """
    # Get size of first frame
    width, height = frames[0].size

    # Initialize video writer
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Write each frame
    for frame in frames:
        # Convert to BGR
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Write to video
        video.write(frame)

    # Release video writer
    video.release()
