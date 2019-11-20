"""
Author: Shehzeen Hussain
"""

import os
import argparse
from os.path import join
import numpy

def main():
    """
    This file automates the running of various experiments 
    for different attacks to generate adversarial videos and
    detections on the newly generated adversarial video using specified detector model.
    Keeps track of results of attacks on different detector models
    and the dataset used for attack.

    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--model_dir', '-mdir', type=str, 
        default="/data2/paarth/faceforensics++_models_subset/face_detection/xception") #dir contains all_c23.p etc
    p.add_argument('--data_dir', '-data', type=str, 
        default="/data2/paarth/DeepFakeDataset/manipulated_sequences/") # dir containing face2face etc
    p.add_argument('--exp_folder', '-exp', type=str, 
        default="/data2/paarth/DFExperiments") # where sub directories will be created
    p.add_argument('--model_type', '-mtype', type=str, default=None) #c23, c40 or raw
    p.add_argument('--faketype', type=str, default=None) # face2face, neural textures etc
    p.add_argument('--attack', '-a', type=str, default="iterative_fgsm")
    p.add_argument('--compress', action='store_true')
    p.add_argument('--cuda', action='store_true')
    

    args = p.parse_args()
    experiment_path = args.exp_folder
    fake_dir = args.faketype
    model_type = args.model_type
    model_dir = args.model_dir
    attack_type = args.attack
    data_dir_path = args.data_dir
    compress = args.compress
    cuda_run = args.cuda

    assert attack_type in ["iterative_fgsm", "robust", "carlini_wagner"]
    assert fake_dir in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    assert model_type in ["c23", "c40", "raw"]

    input_folder_path = join(data_dir_path,fake_dir,model_type,"videos")
    assert os.path.isdir(input_folder_path)

    model_path = join(model_dir,"all_{}.p".format(model_type))
    assert os.path.exists(model_path)

    adversarial_folder_path = join(experiment_path,fake_dir,model_type,"adv_{}".format(attack_type))

    detected_folder_path = join(experiment_path,fake_dir,model_type,"adv_{}_detected".format(attack_type))


    if not os.path.isdir(adversarial_folder_path):
        os.makedirs(adversarial_folder_path)


    if not os.path.isdir(detected_folder_path):
        os.makedirs(detected_folder_path)


    string_command_attack = "python3 attack.py -i {} -mi {} -o {}".format(input_folder_path,model_path,adversarial_folder_path)

    if cuda_run:
        string_command_attack += " --cuda"

    if compress:
        string_command_attack += " --compress"

    string_command_detect = "python3 detect_from_video.py -i {} -mi {} -o {}".format(adversarial_folder_path,model_path,detected_folder_path)

    if cuda_run:
        string_command_detect += " --cuda"

    
    print (">>>>>>>>>>>>>>>>>>>>>>>>Starting Attack")
    print(string_command_attack)
    os.system(string_command_attack)
    print ("<<<<<<<<<<<<<<<<<<<<<<<<<Attack Done") 
    
    print (">>>>>>>>>>>>>>>>>>>>>>>>Starting Detection")
    print(string_command_detect)
    os.system(string_command_detect)
    print ("Done")


if __name__ == '__main__':
    main()
