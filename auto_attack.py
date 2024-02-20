import json
import math
import os

import numpy as np
from autoattack import AutoAttack
from torch import nn
from tqdm import tqdm
from torchvision import transforms

from attack import preprocess_image, get_boundingbox, un_preprocess_image, predict_with_model_legacy
import cv2
import dlib
import torch
from dataset.transform import EfficientNetB4ST_default_data_transforms, xception_default_data_transforms
from threading import Thread, Event
from queue import Queue
from os.path import join

from network.models import model_selection


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


class VideoLoader(Thread):
    def __init__(self, video_path, model_type, size, images_queue: Queue, faces_queue: Queue, bb_queue: Queue,
                 end_event: Event, batch_size: int, transform=None):
        super(VideoLoader, self).__init__()
        # self.video_path = video_path
        self.transform = transform
        self.size = size
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.face_detector = dlib.get_frontal_face_detector()
        self.model_type = model_type
        self.images_queue = images_queue
        self.faces_queue = faces_queue
        self.bb_queue = bb_queue
        self.batch_size = batch_size
        self.end_event = end_event
        self.frame_count = 0

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        while self.frame_count < self.__len__():
            faces_list = []
            images_list = []
            bb = []
            for i in range(self.batch_size):
                ret, image = self.cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector(gray, 1)
                height, width = image.shape[:2]

                if len(faces):
                    # For now only take biggest face
                    face = faces[0]
                    # Face crop with dlib and bounding box scale enlargement
                    x, y, size = get_boundingbox(face, width, height)
                    cropped_face = image[y:y + size, x:x + size]
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    cropped_face = EfficientNetB4ST_default_data_transforms['to_tensor'](cropped_face)
                    cropped_face = cropped_face.unsqueeze(0)
                    faces_list.append(cropped_face)
                    bb.append((x, y, size))
                    images_list.append(image)
                self.frame_count += 1
            self.images_queue.put(images_list)
            self.faces_queue.put(faces_list)
            self.bb_queue.put(bb)
        self.end_event.set()
        # return torch.cat(images_list), torch.cat(faces_tensor), torch.cat(bb)


def un_preprocess_image(image: torch.Tensor):
    numpy_image = image.detach().cpu().numpy()
    numpy_image = numpy_image.transpose(1, 2, 0)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    unprocessed_image1 = np.array((cv2_image * 255) + 0.5, dtype=np.uint8)
    return unprocessed_image1


def preprocess_image(image: np.array, model_type: str):
    unprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if model_type == 'EfficientNetB4ST':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == 'xception':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    unprocessed_image = trans(unprocessed_image).unsqueeze(0).cuda()
    return unprocessed_image


def video_writer(output_path: str, attack_name: str, frames_queue: Queue, attacked_faces_queue: Queue, end_event: Event,
                 model_type: str, fps: int, video_path: str, bb_queue: Queue):
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    file_path = join(output_path, video_fn)
    out = None
    metrics = {
        'total_fake_frames': 0,
        'total_real_frames': 0,
        'total_frames': 0,
        'percent_fake_frames': 0,
        'probs_list': [],
    }
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    if model_type == 'xception':
        model_path = 'models/all_c23.p'
        model = torch.load(model_path)
    elif model_type == 'EfficientNetB4ST':
        model_path = 'models/EfficientNetB4ST_FFPP/bestval.pth'
        model = model_selection('EfficientNetB4ST', 1)
        weights = torch.load(model_path)
        model.load_state_dict(weights)
    m = model.to(device).eval()
    while not end_event.is_set() or frames_queue.qsize() > 0:
        attacked_batch = attacked_faces_queue.get()
        images_batch = frames_queue.get()
        bb_batch = bb_queue.get()
        if out is None:
            height, width = images_batch[0].shape[:2]
            out = cv2.VideoWriter(file_path, fourcc, fps, (height, width)[::-1])
        for frame, perturbed_image, (x, y, bb_size) in zip(images_batch, attacked_batch[attack_name][0], bb_batch):
            original_face = frame[y:y + bb_size, x:x + bb_size]
            unsqueezed_image = perturbed_image.unsqueeze(0)
            # print(f'pert image prediction: {nn.Softmax(dim=1)(ModelWrapper(model, model_type)(unsqueezed_image))}')

            # unpreprocess
            unprocessed_image = un_preprocess_image(perturbed_image)

            test = preprocess_image(unprocessed_image, model_type)
            test_output = m(test)
            output = nn.Softmax(dim=1)(test_output).detach().cpu().numpy().tolist()
            prediction = output[0].index(max(output[0]))
            print('Prediction:', prediction, 'Output:', output)

            metrics['total_frames'] += 1
            label = 'fake' if prediction == 1 else 'real'
            metrics['total_fake_frames'] += 1 if label == 'fake' else 0
            metrics['total_real_frames'] += 1 if label == 'real' else 0
            metrics['probs_list'].append(output[0])
            frame[y:y + bb_size, x:x + bb_size] = unprocessed_image
            out.write(frame)
    out.release()
    metrics['percent_fake_frames'] = metrics['total_fake_frames'] / metrics['total_frames']
    with open(file_path.replace(".avi", "_metrics_attack.json"), "w") as f:
        f.write(json.dumps(metrics))


def frame_writer(output_path: str, attack_name: str, frames_queue: Queue, attacked_faces_queue: Queue, end_event: Event,
                 model_type: str, model, fps: int, video_path: str, bb_queue: Queue):
    video_fn = video_path.split('/')[-1].split('.')[0]
    file_path = join(output_path, video_fn)
    metrics = {
        'total_fake_frames': 0,
        'total_real_frames': 0,
        'total_frames': 0,
        'percent_fake_frames': 0,
        'probs_list': [],
    }
    while not end_event.is_set() or frames_queue.qsize() > 0:
        attacked_batch = attacked_faces_queue.get()
        images_batch = frames_queue.get()
        bb_batch = bb_queue.get()
        for frame, perturbed_image, (x, y, bb_size) in zip(images_batch, attacked_batch[attack_name][0], bb_batch):
            torch.save(perturbed_image, file_path + f'_{metrics["total_frames"]}.pt')
            save_image(perturbed_image, file_path + f'_{metrics["total_frames"]}.png')
            unsqueezed_image = perturbed_image.unsqueeze(0)

            unpreprocessed_image_pil, unpreprocessed_image_pil_numpy = un_preprocess_image2(unsqueezed_image, bb_size)
            test2 = nn.Softmax(dim=1)(
                model(EfficientNetB4ST_default_data_transforms['test'](unpreprocessed_image_pil).unsqueeze(0).cuda()))
            prediction, output = predict_with_model_legacy(unpreprocessed_image_pil_numpy, model=model,
                                                           model_type=model_type)
            print('Prediction:', prediction, 'Output:', output)
            metrics['total_frames'] += 1
            label = 'fake' if prediction == 1 else 'real'
            metrics['total_fake_frames'] += 1 if label == 'fake' else 0
            metrics['total_real_frames'] += 1 if label == 'real' else 0
            metrics['probs_list'].append(output[0])
    metrics['percent_fake_frames'] = metrics['total_fake_frames'] / metrics['total_frames']
    with open(file_path + "_metrics_attack.json", "w") as f:
        f.write(json.dumps(metrics))


def norm_epsilon(eps, preprocess_function, resize_size):
    max_tensor = torch.ones([1, 3, resize_size, resize_size])
    min_tensor = torch.zeros([1, 3, resize_size, resize_size])
    max_eps = float(preprocess_function(max_tensor).max())
    min_eps = float(preprocess_function(min_tensor).min())
    return eps * (max_eps - min_eps)


if __name__ == '__main__':
    videos_path = 'newDataset/Test/fake/Deepfakes/'
    model_type = 'EfficientNetB4ST'
    attack_name = 'square'
    deepfake_type = 'Deepfakes'
    output_path = f'newDataset/Test/attacked/{attack_name}/{model_type}/{deepfake_type}'
    batch_size = 1
    use_cuda = True
    eps = 16 / 255
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    if model_type == 'xception':
        face_size = 299
        model_path = 'models/all_c23.p'
        model = torch.load(model_path)
        p_init = 0.3
    elif model_type == 'EfficientNetB4ST':
        face_size = 224
        model_path = 'models/EfficientNetB4ST_FFPP/bestval.pth'
        model = model_selection('EfficientNetB4ST', 1)
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        p_init = 0.3
    model_legacy = model.to(device).eval()
    model = ModelWrapper(model_legacy, model_type=model_type).eval()
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard')
    adversary.attacks_to_run = [attack_name]
    adversary.verbose = True
    adversary.square.n_queries = 10000
    adversary.square.n_restarts = 1
    adversary.square.verbose = True
    adversary.square.p_init = p_init

    os.makedirs(output_path, exist_ok=True)
    videos = os.listdir(videos_path)

    for video in videos:
        video_path = join(videos_path, video)
        if os.path.exists(os.path.join(output_path, os.path.splitext(video)[0] + '_metrics_attack.json')):
            print(f'Adversarial video already exists for {video}')
            continue
        images_queue = Queue()
        faces_queue = Queue()
        adv_faces_queue = Queue()
        bb_queue = Queue()
        end_event = Event()
        process_end_event = Event()

        video_batch_thread = VideoLoader(video_path, model_type, face_size, images_queue, faces_queue, bb_queue,
                                         end_event, batch_size, preprocess_image, )
        video_batch_thread.start()
        fps = video_batch_thread.fps
        video_writer_thread = Thread(target=video_writer,
                                     args=(output_path, attack_name, images_queue,
                                           adv_faces_queue, process_end_event, model_type, int(fps), video_path, bb_queue,))
        video_writer_thread.start()
        # frame_writer_thread = Thread(target=frame_writer,
        #                              args=(output_path, attack_name, images_queue,
        #                                    adv_faces_queue, process_end_event, model_type, model_legacy,
        #                                    int(fps), video_path, bb_queue,))
        # frame_writer_thread.start()

        number_of_batches = math.ceil(len(video_batch_thread) / batch_size)
        batch_bar = tqdm(total=number_of_batches)
        while not end_event.is_set() or faces_queue.qsize() > 0:
            faces_batch = faces_queue.get()[0]
            labels = torch.ones(len(faces_batch), dtype=torch.long).to(device)
            dict_adv = adversary.run_standard_evaluation_individual(faces_batch.cuda(), labels, bs=len(faces_batch),
                                                                    return_labels=True)

            dict_adv[attack_name] = (
                dict_adv[attack_name][0].clone().detach(), dict_adv[attack_name][1].clone().detach())
            adv_faces_queue.put(dict_adv)
            print(f'faces_queue: {faces_queue.qsize()}, adv_faces_queue: {adv_faces_queue.qsize()}, '
                  f'bb_queue: {bb_queue.qsize()}, images_queue: {images_queue.qsize()}')
            batch_bar.update(1)
        batch_bar.close()
        process_end_event.set()
        video_batch_thread.join()
        video_writer_thread.join()
        # frame_writer_thread.join()
        print(f'{video} finished')
