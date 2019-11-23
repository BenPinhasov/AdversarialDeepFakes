"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os, sys
import argparse
from os.path import join
import cv2
from PIL import Image as pil_image
from tqdm import tqdm
import json


# I don't recommend this, but I like clean terminal output.
import warnings
warnings.filterwarnings("ignore")


def convert_to_mjpeg(video_path,  output_path,
                            start_frame=0, end_frame=None):
    """
    Saves a video as mjpeg
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    metrics_file_source = video_path.replace(".avi", "_metrics_attack.json")
    metrics_fn = video_fn.replace(".avi", "_metrics_attack.json")

    os.makedirs(output_path, exist_ok=True)


    # if metrics exist for the source file, copy them to the dest folder
    if os.path.exists(metrics_file_source):
        with open(metrics_file_source) as sf:
            metrics = json.loads(sf.read())
            metrics_file_dest = join(output_path, metrics_fn)
            with open(metrics_file_dest, "w") as df:
                df.write(json.dumps(metrics))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        
        # Image size
        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])
        writer.write(image)
        pbar.update(1)
    pbar.close()

    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        convert_to_mjpeg(**vars(args))
    else:
        videos = os.listdir(video_path)
        videos = [ video for video in videos if (video.endswith(".mp4") or video.endswith(".avi")) ]
        pbar_global = tqdm(total=len(videos))
        for video in videos:
            args.video_path = join(video_path, video)
            blockPrint()
            try:
                convert_to_mjpeg(**vars(args))
            except:
                pass
            enablePrint()
            pbar_global.update(1)
        pbar_global.close()