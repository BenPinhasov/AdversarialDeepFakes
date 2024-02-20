import argparse
import time

import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import dlib
import torch
from tqdm import tqdm

from dataset.transform import EfficientNetB4ST_default_data_transforms, xception_default_data_transforms
from detect_from_video import predict_with_model, get_boundingbox, preprocess_image

torch.cuda.empty_cache()
import torch.nn as nn
from torchvision import transforms
import os
from os.path import join
import sys
from PIL import Image as pil_image
import json
from network.models import model_selection
from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Saliency, visualization
from matplotlib import pyplot as plt, cm
import numpy as np
from attack import un_preprocess_image
from multiprocessing import Process, Queue, Event


def image_to_display(img, label=None, confidence=None, target=None):
    # img -= img.min()
    # img /= img.max()
    # img = (img * 255).squeeze().T.cpu().detach().numpy()  # get normalized
    # img = np.swapaxes(img, 1, 0).astype(dtype=np.uint8)
    # img = tensor_to_numpy(img).astype(dtype=np.uint8) * 255
    if label and confidence and target is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (5, 15)
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        img = cv2.putText(img.copy(), f'Classification: {label} {round(confidence, 3)} XAI_target: {target}', org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
    else:
        pass
    return img


def display_images(xai_image, face):
    rows_rgb, cols_rgb, channels = face.shape
    rows_gray, cols_gray, _ = xai_image.shape

    rows_comb = max(rows_rgb, rows_gray)
    cols_comb = cols_rgb + cols_gray
    comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

    comb[:rows_rgb, :cols_rgb] = face
    comb[:rows_gray, cols_rgb:] = xai_image

    cv2.imshow('xai_wb', comb)
    cv2.waitKey(1)
    return comb


def load_model(model_type: str, model_path: str, cuda: bool):
    if model_path is not None:
        post_function = nn.Softmax(dim=1)
        if not cuda:
            model = torch.load(model_path, map_location="cpu")
        else:
            if model_type == 'meso':
                # onnx_model = onnx.load(model_path)
                # model =
                # model = ConvertModel(onnx_model)
                model = model_selection(model_type, 2)[0]
                weights = torch.load(model_path)
                model.load_state_dict(weights)
            elif model_type == 'xception':
                model = torch.load(model_path)
            elif model_type == 'mymeso':
                model = model_selection(model_type, 2)[0]
                weights = torch.load(model_path)
                model.load_state_dict(weights)
            elif model_type == 'EfficientNetB4ST':
                model = model_selection('EfficientNetB4ST', 2)
                weights = torch.load(model_path)
                model.load_state_dict(weights)
                post_function = nn.Softmax(dim=1)
            else:
                raise f"{model_type} not supported"
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        print("Converting mode to cuda")
        model = model.eval().cuda()
        for param in model.parameters():
            param.requires_grad = True
        print("Converted to cuda")
    return model, post_function


def to_gray_image(x):
    x -= x.min()
    x /= x.max() + np.spacing(1)
    x *= 255
    return np.array(x, dtype=np.uint8)


def video_writer_process(output_path, model_type, xai_method, video_fn, writer_dim, fps, frames_queue,
                         close_event: Event):
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # writer = cv2.VideoWriter(os.path.join(output_path, xai_method, video_fn), fourcc, fps,
    #                          writer_dim)
    print(f'video writer process for {xai_method} method and video file name {video_fn} started')
    video_fn = video_fn.replace(".avi", "")
    while not close_event.is_set():
        if not frames_queue.empty():
            xai_img, cropped_face, frame_id = frames_queue.get()
            # writer.write(frame)
            if not os.path.exists(os.path.join(output_path, model_type, f'original/{video_fn}_{frame_id}.jpg')):
                cv2.imwrite(os.path.join(output_path, model_type, f'original/{video_fn}_{frame_id}.jpg'), cropped_face)
            if not os.path.exists(os.path.join(output_path, model_type, xai_method, f'{video_fn}_{frame_id}.jpg')):
                cv2.imwrite(os.path.join(output_path, model_type, xai_method, f'{video_fn}_{frame_id}.jpg'), xai_img)
        else:
            time.sleep(0.1)
    # writer.release()
    print(f'video writer process for {xai_method} method and video file name {video_fn} finished')


def preprocess_image_square(image: np.array, model_type: str):
    # unprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    unprocessed_image = image
    if model_type == 'EfficientNetB4ST':
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == 'xception':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    unprocessed_image = trans(unprocessed_image).unsqueeze(0).cuda()
    return unprocessed_image


class ModelWrapper(nn.Module):
    def __init__(self, model, model_type):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, x):
        if self.model_type == 'EfficientNetB4ST':
            resized_image = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)
            normalized_image = EfficientNetB4ST_default_data_transforms['normalize'](resized_image)
        elif self.model_type == 'xception':
            resized_image = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
            normalized_image = xception_default_data_transforms['normalize'](resized_image)
        return self.model(normalized_image)


def predict_with_model_square(processed_image, model, post_function=None):
    output = model(processed_image)
    if post_function is not None:
        output = post_function(output)
        prediction = torch.argmax(output)
        prediction = int(prediction.cpu().numpy())
    output = output.detach().cpu().numpy().tolist()
    return prediction, output


def compute_attr(video_path, model_path, model_type, output_path, xai_methods, cuda):
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    video_fn = os.path.basename(video_path).split('.')[0] + '.avi'
    is_square = video_path.find('square') != -1
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    if model_type == 'EfficientNetB4ST':
        writer_dim = (224, 224)
    else:
        writer_dim = (299, 299)

    face_detector = dlib.get_frontal_face_detector()
    model_legacy, post_function = load_model(model_type, model_path, cuda)
    model = ModelWrapper(model_legacy, model_type)
    # Frame numbers and length of output video
    start_frame = 0
    end_frame = None
    frame_num = 0
    no_face_count = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)
    xai_metrics = {}
    xai = {}
    writers_process = {}
    writers_process_events = {}
    writers_process_frames_queues = {}
    for xai_method in xai_methods:
        os.makedirs(output_path + f'/{model_type}/{xai_method}', exist_ok=True)
        os.makedirs(output_path + f'/{model_type}/original', exist_ok=True)
        xai[xai_method] = eval(f'{xai_method}')(model)
        if not os.path.exists(os.path.join(output_path, xai_method, video_fn.replace(".avi", "_metrics.json"))):
            writers_process_frames_queues[xai_method] = Queue()
            writers_process_events[xai_method] = Event()
            p = Process(target=video_writer_process,
                        args=(output_path, model_type, xai_method, video_fn, writer_dim, fps,
                              writers_process_frames_queues[xai_method],
                              writers_process_events[xai_method],))
            writers_process[xai_method] = p
            p.start()
        xai_metrics[xai_method] = {
            'total_frames': 0,
            'no_face_count': 0,
            'total_fake_predictions': [],
            'total_real_predictions': [],
            'prediction_list': [],
            'probs_list': []
        }
    break_flag = False
    frame_id = 0
    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break
        frame_num += 1
        if frame_num < start_frame:
            continue
        pbar.update(1)
        height, width = image.shape[:2]

        # 2. find faces with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]
        else:
            no_face_count += 1
            continue
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
        if is_square:
            preprocessed_image = preprocess_image_square(cropped_face, model_type)
            prediction, output = predict_with_model_square(preprocessed_image, model, post_function=post_function)
        else:
            preprocessed_image = preprocess_image(cropped_face, model_type)
            prediction, output = predict_with_model(cropped_face, model, model_type, post_function=post_function,
                                                    cuda=cuda)
        print(f'prediction: {prediction} output: {output}')
        if prediction == 1:
            continue
        for xai_method in xai_methods:
            if os.path.exists(
                    os.path.join(output_path, model_type, xai_method, video_fn.replace(".avi", "_metrics.json"))):
                print(f'metric file for {video_fn} exists. continue')
                reader.release()
                break_flag = True
                continue
            if break_flag:
                break
            if xai_method == 'IntegratedGradients':
                xai_img = xai[xai_method].attribute(preprocessed_image, target=prediction, internal_batch_size=1)
            else:
                xai_img = xai[xai_method].attribute(preprocessed_image, target=prediction)

            xai_img = un_preprocess_image(xai_img, xai_img.shape[2])
            xai_metrics[xai_method]['prediction_list'].append(prediction)
            xai_metrics[xai_method]['total_frames'] += 1.
            xai_metrics[xai_method]['probs_list'].append(output[0])
            # writers[xai_method].write(xai_img)
            writers_process_frames_queues[xai_method].put_nowait((xai_img, cropped_face, frame_id))
        frame_id += 1
    pbar.close()
    for xai_method in xai_methods:
        sum_of_fakes = sum(xai_metrics[xai_method]['prediction_list'])
        len_predictions = len(xai_metrics[xai_method]['prediction_list'])
        xai_metrics[xai_method]['total_fake_predictions'] = sum_of_fakes
        xai_metrics[xai_method]['total_real_predictions'] = len_predictions - sum_of_fakes
        xai_metrics[xai_method]['no_face_count'] = no_face_count
        if not os.path.exists(os.path.join(output_path, xai_method, video_fn.replace(".avi", "_metrics.json"))):
            with open(os.path.join(output_path, model_type, xai_method, video_fn.replace(".avi", "_metrics.json")),
                      "w") as f:
                f.write(json.dumps(xai_metrics[xai_method]))
            print(f'Finished! Output saved under {os.path.join(output_path, xai_method)}')
            # writers[xai_method].release()
            writers_process_events[xai_method].set()
            writers_process[xai_method].join()


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--model_type', '-mt', type=str, default="xception")
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--xai_methods', '-x', nargs='*', type=str)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    video_path = args.video_path
    # args.xai_methods = ['GuidedBackprop', 'Saliency', 'InputXGradient', 'IntegratedGradients']
    # videos = []
    # for root, directories, files in os.walk(video_path):
    #     for filename in files:
    #         # Join the two strings in order to form the full filepath
    #         filepath = os.path.join(root, filename)
    #         # Check if it is a mp4 file
    #         if filepath.endswith('.mp4') or filepath.endswith('.avi'):
    #             videos.append(filepath)
    videos = os.listdir(video_path)
    videos = [video for video in videos if (video.endswith(".mp4") or video.endswith(".avi"))]
    pbar_global = tqdm(total=len(videos))
    for video in videos:
        args.video_path = join(video_path, video)
        compute_attr(**vars(args))
        pbar_global.update(1)
    pbar_global.close()

    # # video_path = 'temadv/Zelensky_deepfake_adv.avi'
    # video_path = 'Data/Zelensky_deepfake.mp4'
    # model_type = 'meso'
    # model_path = 'faceforensics++_models_subset/face_detection/Meso/Meso4_deepfake.pkl'
    # cuda = True
    # xai_method = 'Saliency'  # IntegratedGradients, InputXGradient, GuidedBackprop, Saliency
    # xai_methods = ['GuidedBackprop', 'Saliency', 'InputXGradient', 'IntegratedGradients']
    # # xai_methods = ['GuidedBackprop']
    # target = 1
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fps = 10
    #
    # for xai_method in xai_methods:
    #     try:
    #         # fix video path string and get video file name (video_fn)
    #         video_path = video_path.replace('\\', '/') if '\\' in video_path else video_path
    #         video_fn = video_path.split('/')[-1].split('.')[0]
    #         # create video writer
    #         # writer = cv2.VideoWriter(video_fn + '_xai_' + xai_method + '.avi', fourcc, fps, (299, 598)[::-1])
    #         writer = cv2.VideoWriter(video_fn + '_xai_' + xai_method + '_target_' + str(target) + '.avi', fourcc, fps,
    #                                  (600, 600)[::-1])
    #         # create frame reader
    #         reader = cv2.VideoCapture(video_path)
    #
    #         # create face detector
    #         face_detector = dlib.get_frontal_face_detector()
    #
    #         # load model
    #         if model_path is not None:
    #             if not cuda:
    #                 model = torch.load(model_path, map_location="cpu")
    #             else:
    #                 if model_type == 'meso':
    #                     model = model_selection(model_type, 2)[0]
    #                     weights = torch.load(model_path)
    #                     model.load_state_dict(weights)
    #                 elif model_type == 'xception':
    #                     model = torch.load(model_path)
    #                 else:
    #                     raise f"{model_type} not supported"
    #             print('Model found in {}'.format(model_path))
    #         else:
    #             raise 'No model found'
    #         if cuda:
    #             print("Converting mode to cuda")
    #             model = model.cuda()
    #             for param in model.parameters():
    #                 param.requires_grad = True
    #             print("Converted to cuda")
    #
    #         # create xai
    #         if xai_method == 'IntegratedGradients':
    #             xai = IntegratedGradients(model)
    #         elif xai_method == 'InputXGradient':
    #             xai = InputXGradient(model)
    #         elif xai_method == 'GuidedBackprop':
    #             xai = GuidedBackprop(model)
    #         elif xai_method == 'Saliency':
    #             xai = Saliency(model)
    #         else:
    #             raise 'No xai method chosen'
    #         while reader.isOpened():
    #             ret, image = reader.read()
    #             if not ret:
    #                 break
    #             # get image size
    #             height, width = image.shape[:2]
    #             # detect face
    #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             faces = face_detector(gray, 1)
    #             if len(faces):
    #                 # take only the biggest face
    #                 face = faces[0]
    #                 x, y, size = get_boundingbox(face, width, height)
    #                 cropped_face = image[y:y + size, x:x + size, :]
    #
    #                 preprocessed_image = preprocess_image(cropped_face, model_type, cuda)
    #
    #                 output = model(preprocessed_image)
    #                 output = nn.Softmax(dim=1)(output)
    #                 confidence, prediction = torch.max(output, 1)
    #                 confidence = float(confidence[0])
    #                 pred = int(prediction.cpu().numpy())
    #                 label = 'fake' if pred == 1 else 'real'
    #                 # face_to_disp = image_to_display(preprocessed_image, label, confidence)
    #                 # face_to_disp = image_to_display(preprocessed_image)
    #                 face_to_disp = tensor_to_numpy(preprocessed_image)
    #                 if xai_method == 'IntegratedGradients':
    #                     xai_img = xai.attribute(preprocessed_image, target=target, internal_batch_size=1)
    #                 else:
    #                     xai_img = xai.attribute(preprocessed_image, target=target)
    #                 # xai_img = image_to_display(xai_img)
    #                 xai_img = tensor_to_numpy(xai_img)
    #                 # fig, ax = visualization.visualize_image_attr_multiple(xai_img, face_to_disp,
    #                 #                                                       methods=['heat_map', 'original_image'],
    #                 #                                                       signs=['absolute_value', 'absolute_value'],
    #                 #                                                       show_colorbar=True)
    #                 fig, ax = visualization.visualize_image_attr(xai_img, face_to_disp, method='blended_heat_map',
    #                                                              sign='absolute_value', show_colorbar=True,
    #                                                              use_pyplot=False)
    #                 canvas = FigureCanvas(fig)
    #                 canvas.draw()
    #                 buf = canvas.buffer_rgba()
    #                 img_to_disp = image_to_display(np.asarray(buf)[:, :, :3], label=label, confidence=confidence,
    #                                                target=target)
    #                 cv2.imshow(xai_method, img_to_disp)
    #                 cv2.waitKey(1)
    #                 # xai_gray = cv2.cvtColor(xai_img, cv2.COLOR_BGR2GRAY)
    #                 # (thresh, blackAndWhiteImage) = cv2.threshold(xai_gray, 127, 255, cv2.THRESH_BINARY)
    #                 # TODO: Apply color map https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
    #                 # blackAndWhiteImage = cv2.applyColorMap(xai_gray, cv2.COLORMAP_HOT)
    #                 # comb_image = display_images(blackAndWhiteImage, cv2.cvtColor(face_to_disp, cv2.COLOR_RGB2BGR))
    #                 writer.write(img_to_disp)
    #                 pass
    #     except:
    #         import traceback
    #         traceback.print_exc()
    #         pass
    #     writer.release()
    # pass


if __name__ == '__main__':
    main()
