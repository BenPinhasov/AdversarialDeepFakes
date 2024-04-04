import time

import numpy as np
from torch import autograd
import torch
import torch.nn as nn

from attack import un_preprocess_image, calculate_xai_map, check_attacked
from dataset.transform import xception_default_data_transforms, mesonet_default_data_transforms, \
    EfficientNetB4ST_default_data_transforms
import robust_transforms as rt
import random

from zoo_l2_attack_black import l2_attack


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
    elif model_type == "meso":
        resized_image = nn.functional.interpolate(preprocessed_image, size=(256, 256), mode="bilinear",
                                                  align_corners=True)
        norm_transform = mesonet_default_data_transforms['normalize']
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


def robust_fgsm(input_img, model, model_type, cuda=True,
                max_iter=100, alpha=1 / 255.0,
                eps=16 / 255.0, desired_acc=0.95,
                transform_set={"gauss_noise", "gauss_blur", "translation", "resize"}
                ):
    def _get_transforms(apply_transforms={"gauss_noise", "gauss_blur", "translation", "resize"}):

        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda=cuda),
            ]

        if "gauss_blur" in apply_transforms:
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size=(5, 5), sigma=(5., 5.), cuda=cuda),
                lambda x: rt.gaussian_blur(x, kernel_size=(5, 5), sigma=(10., 10.), cuda=cuda),
                lambda x: rt.gaussian_blur(x, kernel_size=(7, 7), sigma=(5., 5.), cuda=cuda),
                lambda x: rt.gaussian_blur(x, kernel_size=(7, 7), sigma=(10., 10.), cuda=cuda),
            ]

        if "translation" in apply_transforms:
            transform_list += [
                lambda x: rt.translate_image(x, 10, 10, cuda=cuda),
                lambda x: rt.translate_image(x, 10, -10, cuda=cuda),
                lambda x: rt.translate_image(x, -10, 10, cuda=cuda),
                lambda x: rt.translate_image(x, -10, -10, cuda=cuda),
                lambda x: rt.translate_image(x, 20, 20, cuda=cuda),
                lambda x: rt.translate_image(x, 20, -20, cuda=cuda),
                lambda x: rt.translate_image(x, -20, 10, cuda=cuda),
                lambda x: rt.translate_image(x, -20, -20, cuda=cuda),
            ]

        if "resize" in apply_transforms:
            transform_list += [
                lambda x: rt.compress_decompress(x, 0.1, cuda=cuda),
                lambda x: rt.compress_decompress(x, 0.2, cuda=cuda),
                lambda x: rt.compress_decompress(x, 0.3, cuda=cuda),
            ]

        return transform_list

    input_var = autograd.Variable(input_img, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0

    loss_criterion = nn.CrossEntropyLoss()

    while iter_no < max_iter:
        transform_functions = _get_transforms(transform_set)

        loss = 0

        all_fooled = True
        print("**** Applying Transforms ****")
        for transform_fn in transform_functions:

            transformed_img = transform_fn(input_var)
            prediction, output, logits = predict_with_model(transformed_img, model, model_type, cuda=cuda)

            if output[0][0] < desired_acc:
                all_fooled = False
            loss += torch.clamp(logits[0][1] - logits[0][0] + 10, min=0.0)
            # loss += loss_criterion(logits, target_var)

        print("*** Finished Transforms **, all fooled", all_fooled)
        if all_fooled:
            break

        loss /= (1. * len(transform_functions))
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


def carlini_wagner_attack(input_img, model, model_type, cuda=True,
                          max_attack_iter=500, alpha=0.005,
                          const=1e-3, max_bs_iter=5, confidence=20.0):
    def torch_arctanh(x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    if model_type == 'EfficientNetB4ST':
        post_function = nn.Sigmoid()
    else:
        post_function = nn.Softmax(dim=1)
    attack_w = autograd.Variable(torch_arctanh(input_img.data - 1), requires_grad=True)
    bestl2 = 1e10
    bestscore = -1

    lower_bound_c = 0
    upper_bound_c = 1.0
    bestl2 = 1e10
    bestimg = None
    optimizer = torch.optim.Adam([attack_w], lr=alpha)
    for bsi in range(max_bs_iter):
        for iter_no in range(max_attack_iter):
            adv_image = 0.5 * (torch.tanh(input_img + attack_w) + 1.)
            prediction, output, logits = predict_with_model(adv_image, model, model_type, cuda=cuda,
                                                            post_function=post_function)
            # if model_type == 'EfficientNetB4ST':
            #     logits = nn.Softmax(dim=1)(logits)
            loss1 = torch.clamp(logits[0][1] - logits[0][0] + confidence, min=0.0)
            loss2 = torch.norm(adv_image - input_img, 2)

            loss_total = loss2 + const * loss1
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if iter_no % 50 == 0:
                print("BSI {} ITER {}".format(bsi, iter_no), output)
                print("Losses", loss_total, loss1.data, loss2)

        # binary search for const
        if (logits[0][0] - logits[0][1] > confidence):
            if loss2 < bestl2:
                bestl2 = loss2
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Best l2", bestl2)
                bestimg = adv_image.detach().clone().data

            upper_bound_c = min(upper_bound_c, const)
        else:
            lower_bound_c = max(lower_bound_c, const)

        const = (lower_bound_c + upper_bound_c) / 2.0

    meta_data = {}
    if bestimg is not None:
        meta_data['l2_norm'] = bestl2.detach().item()
        return bestimg, meta_data
    else:
        meta_data['l2_norm'] = loss2.detach().item()
        return adv_image, meta_data


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


def adaptive_black_box_attack(input_img, deepfake_detector_model, deepfake_detector_model_type,
                              attacked_detector_model, xai_calculator, xai_method, crop_size,
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

    def _find_nes_gradient(input_var, transform_functions, deepfake_detector_model, deepfake_detector_model_type,
                           attacked_detector_model, xai_calculator, crop_size, xai_method, num_samples=10, sigma=0.001):
        g = 0
        _num_queries = 0
        for sample_no in range(num_samples):
            # for transform_func in transform_functions:
            rand_noise = torch.randn_like(input_var)
            img1 = input_var + sigma * rand_noise
            img2 = input_var - sigma * rand_noise

            unprocessed_img1 = un_preprocess_image(img1, crop_size)
            unprocessed_img2 = un_preprocess_image(img2, crop_size)
            xai1 = calculate_xai_map(unprocessed_img1, deepfake_detector_model, deepfake_detector_model_type,
                                     xai_calculator, xai_method, cuda=cuda)
            xai2 = calculate_xai_map(unprocessed_img2, deepfake_detector_model, deepfake_detector_model_type,
                                     xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs1, _ = check_attacked((img1), xai1, attacked_detector_model,
                                                       cuda=cuda)
                _, attacked_probs2, _ = check_attacked((img2), xai2, attacked_detector_model,
                                                       cuda=cuda)
                prediction1, probs_1, _ = predict_with_model((img1), deepfake_detector_model,
                                                             deepfake_detector_model_type, cuda=cuda)

                prediction2, probs_2, _ = predict_with_model((img2), deepfake_detector_model,
                                                             deepfake_detector_model_type, cuda=cuda)

            _num_queries += 2
            g = g + (probs_1[0][0] + attacked_probs1[0][0]) * rand_noise
            g = g - (probs_2[0][0] + attacked_probs2[0][0]) * rand_noise
            g = g.data.detach()

            del rand_noise
            del img1
            del prediction1, probs_1
            del prediction2, probs_2
            del attacked_probs1
            del attacked_probs2

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
            with torch.no_grad():
                _, output, _ = predict_with_model(input_var, deepfake_detector_model, deepfake_detector_model_type,
                                                  cuda=cuda)
            unprocessed_img = un_preprocess_image(input_var, crop_size)
            xai = calculate_xai_map(unprocessed_img, deepfake_detector_model, deepfake_detector_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs, _ = check_attacked(input_var, xai, attacked_detector_model, cuda=cuda)
            num_queries += 1
            if (output[0][0] > desired_acc) and (attacked_probs[0][0] > 0.9):
                warm_start_done = True

        if warm_start_done:
            # choose all transform functions
            transform_functions = _get_transforms(transform_set)
        else:
            transform_functions = _get_transforms({})  # returns identity function

        all_fooled = True
        print("Testing transformation outputs", iter_no)
        for transform_fn in transform_functions:
            with torch.no_grad():
                _, output, _ = predict_with_model(transform_fn(input_var), deepfake_detector_model,
                                                  deepfake_detector_model_type, cuda=cuda)
            unprocessed_img = un_preprocess_image(input_var, crop_size)
            xai = calculate_xai_map(unprocessed_img, deepfake_detector_model, deepfake_detector_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs, _ = check_attacked(transform_fn(input_var), xai, attacked_detector_model, cuda=cuda)
            num_queries += 1
            print(output)
            print(attacked_probs)
            if (output[0][0] < desired_acc) or (attacked_probs[0][0] < 0.9):
                all_fooled = False

        print("All transforms fooled:", all_fooled, "Warm start done:", warm_start_done)
        if warm_start_done and all_fooled:
            break

        t_start = time.time()
        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var=input_var,
                                                                            transform_functions=transform_functions,
                                                                            deepfake_detector_model=deepfake_detector_model,
                                                                            deepfake_detector_model_type=deepfake_detector_model_type,
                                                                            attacked_detector_model=attacked_detector_model,
                                                                            xai_calculator=xai_calculator,
                                                                            crop_size=crop_size,
                                                                            xai_method=xai_method)
        print(f'Gradient calculation time: {time.time() - t_start}')
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


def adaptive_black_box_attack_batches(input_img, deepfake_detector_model, deepfake_detector_model_type,
                                      attacked_detector_model, xai_calculator, xai_method, crop_size,
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

    def _find_nes_gradient(input_var, transform_functions, deepfake_detector_model, deepfake_detector_model_type,
                           attacked_detector_model, xai_calculator, crop_size, xai_method, num_samples=10, sigma=0.001):
        g = 0
        _num_queries = 0
        input_shape = input_var.shape[1:]
        rand_noise = torch.randn(num_samples, num_samples, *input_shape, device=input_var.device) * sigma

        imgs1 = input_var.expand(num_samples, -1, -1, -1) + sigma * rand_noise
        imgs2 = input_var.expand(num_samples, -1, -1, -1) - sigma * rand_noise

        unprocessed_imgs1 = un_preprocess_image_batch(imgs1, crop_size)
        unprocessed_imgs2 = un_preprocess_image_batch(imgs2, crop_size)

        xais1 = calculate_xai_map_batch(unprocessed_imgs1, deepfake_detector_model, deepfake_detector_model_type,
                                        xai_calculator, xai_method, cuda=cuda)
        xais2 = calculate_xai_map_batch(unprocessed_imgs2, deepfake_detector_model, deepfake_detector_model_type,
                                        xai_calculator, xai_method, cuda=cuda)

        _, attacked_probs1, _ = check_attacked_batch(img1, xais1, deepfake_detector_model, cuda=cuda)
        _, attacked_probs2, _ = check_attacked_batch(img2, xais2, deepfake_detector_model, cuda=cuda)

        predictions1, probs1 = predict_with_model_batch(imgs1, deepfake_detector_model, deepfake_detector_model_type,
                                                        cuda=cuda)
        prediction2, probs2 = predict_with_model_batch(img2, deepfake_detector_model, deepfake_detector_model_type,
                                                       cuda=cuda)
        _num_queries += 2 * num_samples

        g = ((probs_1[:, :, 0] + attacked_probs1[:, :, 0]) - (
                    probs_2[:, :, 0] + attacked_probs2[:, :, 0])) @ rand_noise.view(batch_size * num_samples, -1)
        g = g.sum(dim=0).data.detach()
        #TODO: Understand what is dims of g one the batch_Size is 1

        for sample_no in range(num_samples):
            # for transform_func in transform_functions:
            rand_noise = torch.randn_like(input_var)
            img1 = input_var + sigma * rand_noise
            img2 = input_var - sigma * rand_noise

            unprocessed_img1 = un_preprocess_image(img1, crop_size)
            unprocessed_img2 = un_preprocess_image(img2, crop_size)
            xai1 = calculate_xai_map(unprocessed_img1, deepfake_detector_model, deepfake_detector_model_type,
                                     xai_calculator, xai_method, cuda=cuda)
            xai2 = calculate_xai_map(unprocessed_img2, deepfake_detector_model, deepfake_detector_model_type,
                                     xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs1, _ = check_attacked((img1), xai1, attacked_detector_model,
                                                       cuda=cuda)
                _, attacked_probs2, _ = check_attacked((img2), xai2, attacked_detector_model,
                                                       cuda=cuda)
                prediction1, probs_1, _ = predict_with_model((img1), deepfake_detector_model,
                                                             deepfake_detector_model_type, cuda=cuda)

                prediction2, probs_2, _ = predict_with_model((img2), deepfake_detector_model,
                                                             deepfake_detector_model_type, cuda=cuda)

            _num_queries += 2
            g = g + (probs_1[0][0] + attacked_probs1[0][0]) * rand_noise
            g = g - (probs_2[0][0] + attacked_probs2[0][0]) * rand_noise
            g = g.data.detach()

            del rand_noise
            del img1
            del prediction1, probs_1
            del prediction2, probs_2
            del attacked_probs1
            del attacked_probs2

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
            with torch.no_grad():
                _, output, _ = predict_with_model(input_var, deepfake_detector_model, deepfake_detector_model_type,
                                                  cuda=cuda)
            unprocessed_img = un_preprocess_image(input_var, crop_size)
            xai = calculate_xai_map(unprocessed_img, deepfake_detector_model, deepfake_detector_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs, _ = check_attacked(input_var, xai, attacked_detector_model, cuda=cuda)
            num_queries += 1
            if (output[0][0] > desired_acc) and (attacked_probs[0][0] > 0.9):
                warm_start_done = True

        if warm_start_done:
            # choose all transform functions
            transform_functions = _get_transforms(transform_set)
        else:
            transform_functions = _get_transforms({})  # returns identity function

        all_fooled = True
        print("Testing transformation outputs", iter_no)
        for transform_fn in transform_functions:
            with torch.no_grad():
                _, output, _ = predict_with_model(transform_fn(input_var), deepfake_detector_model,
                                                  deepfake_detector_model_type, cuda=cuda)
            unprocessed_img = un_preprocess_image(input_var, crop_size)
            xai = calculate_xai_map(unprocessed_img, deepfake_detector_model, deepfake_detector_model_type,
                                    xai_calculator, xai_method, cuda=cuda)
            with torch.no_grad():
                _, attacked_probs, _ = check_attacked(transform_fn(input_var), xai, attacked_detector_model, cuda=cuda)
            num_queries += 1
            print(output)
            print(attacked_probs)
            if (output[0][0] < desired_acc) or (attacked_probs[0][0] < 0.9):
                all_fooled = False

        print("All transforms fooled:", all_fooled, "Warm start done:", warm_start_done)
        if warm_start_done and all_fooled:
            break

        t_start = time.time()
        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var=input_var,
                                                                            transform_functions=transform_functions,
                                                                            deepfake_detector_model=deepfake_detector_model,
                                                                            deepfake_detector_model_type=deepfake_detector_model_type,
                                                                            attacked_detector_model=attacked_detector_model,
                                                                            xai_calculator=xai_calculator,
                                                                            crop_size=crop_size,
                                                                            xai_method=xai_method)
        print(f'Gradient calculation time: {time.time() - t_start}')
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


def l2_black_box_attack(input_img, model, model_type,
                        cuda=True, max_iter=100, alpha=1 / 255.0,
                        eps=16 / 255.0, desired_acc=0.90):
    target_var = autograd.Variable(torch.LongTensor([0])).cuda()
    attack, score = l2_attack(input_img, target_var, model, targeted=True, use_log=True,
                              use_tanh=True, solver="adam", reset_adam_after_found=True, abort_early=True,
                              batch_size=1, max_iter=100, const=0.01, confidence=20.0, early_stop_iters=100,
                              binary_search_steps=2,
                              step_size=0.01, adam_beta1=0.9, adam_beta2=0.999)
    return attack, score
