# Attack Detection On Deepfake Detector

[Paper](https://arxiv.org/abs/2403.02955)

## Setup Requirements

The code is based on PyTorch 11.7 with cuda and requires Python 3.9

Install requirements via ```pip install -r requirements.txt```

## Dataset
To download the FaceForensics++ dataset, you need to fill out their google form and and once accepted, they will send you the link to the download script.

Once, you obtain the download link, please head to the [download section of FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset). You can also find details about the generation of the dataset there. To reproduce the experiment results in this paper, you only need to download the c23 videos of all the fake video generation methods.

## Small subset to try out the attacks
If you want to try out the attacks on a very small subset of this dataset, create a directory `Data/` in the root folder,  download the zipped dataset from [this link](http://adversarialdeepfakes.github.io/dfsubset.zip) and save unzip inside `Data/` to have the following directory structure:

```
Data/
  - DFWebsite/
      - Deepfakes/
      - Face2Face/
      - FaceSwap/
      - NeuralTextures/
```

## Victim Pre-trained Models

### XceptionNet
The authors of FaceForensics++ provide XceptionNet model trained on our FaceForensics++ dataset. 
You can find our used models under [this link](http://kaldir.vc.in.tum.de:/FaceForensics/models/faceforensics++_models.zip). Download zip and unzip it in the root project directory.

### MesoNet

We use the PyTorch implementation of [MesoNet](https://github.com/HongguLiu/MesoNet-Pytorch). The pretrained model can be downloaded from [here](https://github.com/HongguLiu/MesoNet-Pytorch/blob/master/output/Mesonet/best.pkl?raw=true). Once downloaded save the pkl as `Meso4_deepfake.pkl` inside ```faceforensics++_models_subset/face_detection/Meso```  directory which was created by unzipping the XceptionNet models in the previous link. 

After saving the weights the `faceforensics++_models_subset/` directory should have the following structure:

```
faceforensics++_models_subset/
  - face_detection/
    - Meso
      - Meso4_deepfake.pkl
    - xception
      - all_c23.p
```
    


### Running an attack on videos directory

This setup is for running pgd, fgsm, nes attack to create adversarial examples on video files in directory. 
```shell
python attack.py
-i <path to input folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights>
-mt <type of model, choose either xception or EfficientNetB4ST >
-o <path to output folder, will contain output videos >
-a <type of attack, choose from the following: pgd, fgsm, nes >
--eps <epsilon value for the attack >
--cuda < if provided will run the attack on GPU >
```
Example:
```shell
python attack.py -i Data/DFWebsite/DeepFakes/c23/videos/ -mi models/xception.p -mt xception -o temadv/ -a pgd --cuda --eps 0.01
```

This setup is for running apgd, square attack to create adversarial examples on video files in directory.
```shell
python auto_attack.py
-i <path to input folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights>
-mt <type of model, choose either xception or EfficientNetB4ST >
-o <path to output folder, will contain output videos >
-a <type of attack, choose from the following: apgd-ce, square >
--eps <epsilon value for the attack >
--cuda < if provided will run the attack on GPU >
```
Example:
```shell
python auto_attack.py -i Data/DFWebsite/DeepFakes/c23/videos/ -o temadv/ -mt EfficientNetB4ST -mi models/EfficientNetB4ST.pth -a apgd-ce --cuda --eps 0.01
```
### Create XAI maps for test or train set

this setup is for creating XAI maps for the test or train set of videos in a directory.

```shell
python create_xai.py
--video_path <path to directory containing videos>
--model_path <path to deepfake detector model>
--model_type <type of model, choose either xception or EfficientNetB4ST >
--output_path <path to output directory, will contain output frames >
--cuda < if provided will run on GPU >
--xai_methods <list of xai methods to use, choose from the following: GuidedBackprop, Saliency, InputXGradient, IntegratedGradients >
```
Example:
```shell
python create_xai.py --video_path tempadv/attacked/apgd-ce/EfficientNetB4ST --model_path models\EfficientNetB4ST.pth --model_type EfficientNetB4ST --output_path Frames/attacked/<attack_name> --cuda --xai_methods GuidedBackprop Saliency InputXGradient IntegratedGradients
```
The output will be is a directory containing facecrops and directory for each XAI method containing the XAI maps for each frame of the video.

Example of the directory structure of videos detected with EfficientNetB4ST deepfake detector:
```
<output path>/
    - EfficientNetB4ST/
      - Frames/
      - GuidedBackprop/
      - Saliency/
      - InputXGradient/
      - IntegratedGradients/
```
