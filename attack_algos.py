import time

from torch import autograd



from utils import *
import torch
import torch.nn as nn
import robust_transforms as rt
from dataset.transform import xception_default_data_transforms, EfficientNetB4ST_default_data_transforms
import random


def predict_with_model(preprocessed_image, model, model_type, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real), output probs, logits
    """

    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    if model_type == "xception":
        resized_image = nn.functional.interpolate(preprocessed_image, size=(299, 299), mode="bilinear",
                                                  align_corners=True)
        norm_transform = xception_default_data_transforms['normalize']
        normalized_image = norm_transform(resized_image)
    elif model_type == 'EfficientNetB4ST':
        resized_image = nn.functional.interpolate(preprocessed_image, size=(224, 224), mode="bilinear",
                                                  align_corners=True)
        norm_transform = EfficientNetB4ST_default_data_transforms['normalize']
        normalized_image = norm_transform(resized_image)
        # normalized_image = preprocessed_image

    logits = model(normalized_image)
    output = post_function(logits)

    if model_type == 'EfficientNetB4ST':
        fake_pred = output[0][1].item()
        real_pred = 1 - fake_pred
        output = np.array([real_pred, fake_pred])
        prediction = float(np.argmax(output))
        output = [output.tolist()]
    else:
        # Cast to desired
        _, prediction = torch.max(output, 1)  # argmax
        prediction = float(prediction.cpu().numpy())
        output = output.detach().cpu().numpy().tolist()
    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def xai_attack_iterative_fgsm(input_img, deepfake_model, deepfake_model_type, crop_size,
                              xai_calculator, xai_method, cuda=True,
                              max_iter=100, alpha=1 / 255.0, eps=16 / 255.0, desired_acc=0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)
    post_function = nn.Softmax(dim=1)
    first_xai_map = None
    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    while iter_no < max_iter:
        _, output_deepfake_detector, logits_deepfake_detector = predict_with_model(input_var, deepfake_model,
                                                                                   deepfake_model_type,
                                                                                   cuda=cuda,
                                                                                   post_function=post_function)
        unpreprocessed_image = un_preprocess_image(input_var, crop_size)
        xai_map = calculate_xai_map(unpreprocessed_image, deepfake_model, deepfake_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
        if first_xai_map is None:
            first_xai_map = xai_map.detach().clone()

        if (output_deepfake_detector[0][0]) > desired_acc:
            break
        loss_criterion = nn.CrossEntropyLoss()
        loss1 = loss_criterion(logits_deepfake_detector, target_var)
        loss_criterion2 = nn.MSELoss()
        loss2 = loss_criterion2(xai_map, first_xai_map)
        loss = loss1 + loss2
        if input_var.grad is not None:
            input_var.grad.data.zero_()  # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


def adaptive_iterative_fgsm(input_img, deepfake_model, deepfake_model_type, attacked_detector_model, crop_size,
                            xai_calculator, xai_method, cuda=True,
                            max_iter=100, alpha=1 / 255.0, eps=16 / 255.0, desired_acc=0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    post_function = nn.Softmax(dim=1)
    while iter_no < max_iter:
        _, output_deepfake_detector, logits_deepfake_detector = predict_with_model(input_var, deepfake_model,
                                                                                   deepfake_model_type,
                                                                                   cuda=cuda,
                                                                                   post_function=post_function)
        # if deepfake_model_type == 'EfficientNetB4ST':
        #     # repeated = logits.repeat(1, 2)
        #     # repeated[0][0] *= -1
        #     logits = nn.Softmax(dim=1)(logits)
        unpreprocessed_image = un_preprocess_image(input_var, crop_size)
        xai_map = calculate_xai_map(unpreprocessed_image, deepfake_model, deepfake_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
        _, output_attacked_detector, logits_attacked_detector = check_attacked(input_var, xai_map,
                                                                               attacked_detector_model,
                                                                               cuda=cuda,
                                                                               post_function=post_function)

        if (output_deepfake_detector[0][0]) > desired_acc and (
                output_attacked_detector[0][0] > 0.9):
            break
        loss_criterion = nn.CrossEntropyLoss()
        loss1 = loss_criterion(logits_deepfake_detector, target_var)
        loss_criterion2 = nn.CrossEntropyLoss()
        loss2 = loss_criterion2(logits_attacked_detector, target_var)
        loss = loss1 + loss2
        if input_var.grad is not None:
            input_var.grad.data.zero_()  # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


def iterative_fgsm(input_img, model, model_type, cuda=True, max_iter=100, alpha=1 / 255.0, eps=16 / 255.0,
                   desired_acc=0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    if model_type == 'EfficientNetB4ST':
        post_function = nn.Sigmoid()
    else:
        post_function = nn.Softmax(dim=1)
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var, model, model_type, cuda=cuda,
                                                        post_function=post_function)
        if model_type == 'EfficientNetB4ST':
            # repeated = logits.repeat(1, 2)
            # repeated[0][0] *= -1
            logits = nn.Softmax(dim=1)(logits)
        if (output[0][0] - output[0][1]) > desired_acc:
            break

        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
        if input_var.grad is not None:
            input_var.grad.data.zero_()  # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


def black_box_attack(input_img, model, model_type,
                     cuda=True, max_iter=100, alpha=1 / 255.0,
                     eps=16 / 255.0, desired_acc=0.90,
                     transform_set={"gauss_blur", "translation"}):
    def _get_transforms(apply_transforms={"gauss_noise", "gauss_blur", "translation", "resize"}):

        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda=cuda),
            ]

        if "gauss_blur" in apply_transforms:
            kernel_size = random.randint(3, 6)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size - 1
            sigma = random.randint(5, 7)
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size=(kernel_size, kernel_size), sigma=(sigma * 1., sigma * 1.),
                                           cuda=cuda)
            ]

        if "translation" in apply_transforms:
            x_translate = random.randint(-20, 20)
            y_translate = random.randint(-20, 20)

            transform_list += [
                lambda x: rt.translate_image(x, x_translate, y_translate, cuda=cuda),
            ]

        if "resize" in apply_transforms:
            compression_factor = random.randint(4, 6) / 10.0
            transform_list += [
                lambda x: rt.compress_decompress(x, compression_factor, cuda=cuda),
            ]

        return transform_list

    def _find_nes_gradient(input_var, transform_functions, model, model_type, num_samples=20, sigma=0.001):
        g = 0
        _num_queries = 0
        for sample_no in range(num_samples):
            for transform_func in transform_functions:
                rand_noise = torch.randn_like(input_var)
                img1 = input_var + sigma * rand_noise
                img2 = input_var - sigma * rand_noise

                prediction1, probs_1, _ = predict_with_model(transform_func(img1), model, model_type, cuda=cuda)

                prediction2, probs_2, _ = predict_with_model(transform_func(img2), model, model_type, cuda=cuda)

                _num_queries += 2
                g = g + probs_1[0][0] * rand_noise
                g = g - probs_2[0][0] * rand_noise
                g = g.data.detach()

                del rand_noise
                del img1
                del prediction1, probs_1
                del prediction2, probs_2

        return (1. / (2. * num_samples * len(transform_functions) * sigma)) * g, _num_queries

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0

    # give it a warm start by crafting by fooling without any transformations -> easier
    warm_start_done = False
    num_queries = 0
    while iter_no < max_iter:

        if not warm_start_done:
            _, output, _ = predict_with_model(input_var, model, model_type, cuda=cuda)
            num_queries += 1
            if output[0][0] > desired_acc:
                warm_start_done = True

        if warm_start_done:
            # choose all transform functions
            transform_functions = _get_transforms(transform_set)
        else:
            transform_functions = _get_transforms({})  # returns identity function

        all_fooled = True
        print("Testing transformation outputs", iter_no)
        for transform_fn in transform_functions:
            _, output, _ = predict_with_model(transform_fn(input_var), model, model_type, cuda=cuda)
            num_queries += 1
            print(output)
            if output[0][0] < desired_acc:
                all_fooled = False

        print("All transforms fooled:", all_fooled, "Warm start done:", warm_start_done)
        if warm_start_done and all_fooled:
            break

        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var, transform_functions, model,
                                                                            model_type)
        num_queries += _num_grad_calc_queries
        step_adv = input_var.detach() + alpha * torch.sign(step_gradient_estimate.data.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)

        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    print("L infinity norm", l_inf_norm, l_inf_norm * 255.0)

    meta_data = {
        'num_network_queries': num_queries,
        'attack_iterations': iter_no,
        'l_inf_norm': l_inf_norm,
        'l_inf_norm_255': round(l_inf_norm * 255.0)
    }

    return input_var, meta_data


