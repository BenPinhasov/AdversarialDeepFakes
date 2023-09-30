import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from detect_from_video import get_boundingbox
import dlib


def find_face(image, face_detector, resize):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces):
        face = faces[0]
    else:
        return None
    x, y, size = get_boundingbox(face, width, height)
    cropped_face = image[y:y + size, x:x + size]
    cropped_face = cv2.resize(cropped_face, (resize, resize))
    return cropped_face


def main(video_path, model_type, videos_type, output_path, xai_methods):
    if model_type == 'EfficientNetB4ST':
        frame_size = 224
    elif model_type == 'xception':
        frame_size = 298
    face_detector = dlib.get_frontal_face_detector()
    for xai_method in xai_methods:
        if videos_type == 'fake':
            xai_maps_dir = f'Datasets/manipulated_sequences/Deepfakes/c23/detection_videos/{model_type}/xai/{xai_method}/'
        else:
            xai_maps_dir = f'Datasets/manipulated_sequences/Deepfakes/c23/attacked/{model_type}/xai/{xai_method}/'
        output_dir = output_path + f'{model_type}/{xai_method}/{videos_type}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if glob.glob(video_path + '*.mp4'):
            videos = glob.glob(video_path + '*.mp4')
        else:
            videos = glob.glob(video_path + '*.avi')
        if glob.glob(xai_maps_dir + '*.avi'):
            xai_videos = glob.glob(xai_maps_dir + '*.avi')
        else:
            xai_videos = glob.glob(xai_maps_dir + '*.mp4')
        assert len(videos) == len(xai_videos), "The videos and xai videos folder not in same length"
        pbar_global = tqdm(total=len(videos))
        for video, xai_video in zip(videos, xai_videos):
            video_fn, _ = os.path.splitext(os.path.basename(video))
            xai_video_fn, _ = os.path.splitext(os.path.basename(xai_video))
            assert video_fn == xai_video_fn, f"The video {video_fn} and xai video {xai_video_fn} not the same"
            pbar_global.update(1)
            vid_reader = cv2.VideoCapture(video)
            xai_reader = cv2.VideoCapture(xai_video)
            frame_id = 0
            vid_end_frame = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            xai_end_frame = int(xai_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if vid_end_frame != xai_end_frame:
                continue
            pbar = tqdm(total=vid_end_frame)
            while vid_reader.isOpened() and xai_reader.isOpened():
                vid_ret, vid_frame = vid_reader.read()
                xai_ret, xai_vid_frame = xai_reader.read()
                if not vid_ret or not xai_ret:
                    break
                pbar.update(1)
                vid_resized_frame = find_face(vid_frame, face_detector, frame_size)
                if vid_resized_frame is None:
                    continue
                cv2.imwrite(filename=f'{output_dir}{video_fn}_{frame_id}.jpg', img=vid_resized_frame)
                cv2.imwrite(filename=f'{output_dir}{video_fn}_{frame_id}_xai.jpg', img=xai_vid_frame)
                # combined_frame = np.add(vid_resized_frame, xai_vid_frame, dtype=np.uint16)
                # np.save(f'{output_dir}{video_fn}_{frame_id}.npy', combined_frame)

                frame_id += 1
            pass
            pbar.close()
        pbar_global.close()

    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_type', '-mt', type=str, default="xception")
    p.add_argument('--videos_type', '-vt', type=str, default="fake")
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--xai_methods', '-x', nargs='*', type=str)
    args = p.parse_args()
    main(**vars(args))
