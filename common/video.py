import os
import urllib.request
import cv2
from PIL import Image


def get_zipnerf(path="img/zipnerf.mp4"):
    """
    Downloads the ZipNeRF teaser video if it doesn't already exist locally.
    Args:
        path (str): The local path where the video will be saved.
    Returns:
        a cv2.VideoCapture object for the video.

    For reference:
    - zipnerf_livingroom_time = [4, 10]
    - zipnerf_kitchen_time = [10, 12]
    """
    # check if video already exists
    if not os.path.exists(path):
        # download from source
        print("Downloading video...")
        zipnerf_video_url = "https://jonbarron.info/zipnerf/img/teaser.mp4"
        urllib.request.urlretrieve(zipnerf_video_url, path)
        print(f"Video downloaded to {path}.")
    else:
        print(f"Video already available locally at {path}.")

    return cv2.VideoCapture(path)


def extract_frames(video, start_time, spacing_seconds, num_frames):
    """
    Extracts frames from a video starting at a given time and spacing.

    Args:
    - start_time (int): The time in seconds to start extracting frames.
    - spacing_seconds (float): The time in seconds between each frame.
    - num_frames (int): The number of frames to extract.

    Returns:
    - List[PIL.Image]: The extracted frames.
    """
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {fps} FPS")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

    start_frame = int(start_time * fps)
    spacing_frames = int(spacing_seconds * fps)
    cur_frame = start_frame
    video_frames = []

    for i in range(num_frames):
        cur_frame = start_frame + i * spacing_frames
        video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        success, image = video.read()
        if success:
            # filename = f"frame_{i}.jpg"
            # cv2.imwrite(filename, image)
            # print(f"Saved frame {filename}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            video_frames.append(pil_image)
        else:
            print("Failed to retrieve frame.")

    return video_frames
