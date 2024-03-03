import pickle
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    with open('runs_resnet50_nograd_rocs.pkl', 'rb') as f:
        rocs = pickle.load(f)
    efficientROCs = {key: rocs[key] for key in rocs.keys() if 'Efficient' in key}
    xceptionROCs = {key: rocs[key] for key in rocs.keys() if 'xception' in key}
    attacks = ['square', 'apgd-ce', 'black_box', 'ifgs']
    detectors = ['EfficientNetB4ST', 'xception']
    detector_rocs = {detectors[0]: efficientROCs, detectors[1]: xceptionROCs}
    for detector in detectors:
        for attack in attacks:
            plt.figure(f'{detector}_{attack}')
            for k, v in detector_rocs[detector].items():
                if attack in k:
                    r, fpr, tpr, threshold = v
                    fpr = fpr[1].cpu().detach().numpy()
                    tpr = tpr[1].cpu().detach().numpy()
                    label = k.split(f'_')[1]
                    if label == 'GuidedBackprop':
                        label = 'Guid. Backprop'
                    elif label == 'InputXGradient':
                        label = 'Inp. X Grad.'
                    elif label == 'IntegratedGradients':
                        label = 'Int. Gradients'
                    plt.plot(fpr, tpr, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            if attack == 'black_box':
                attack = 'NES'
            elif attack == 'ifgs':
                attack = 'PGD'
            elif attack == 'apgd-ce':
                attack = 'APGD'
            elif attack == 'square':
                attack = 'Square'
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            if detector == 'xception':
                detector_disp = 'XceptionNet'
            else:
                detector_disp = 'EfficientNetB4ST'
            plt.title(f'{detector_disp} - {attack}')
            plt.legend()
            # plt.show(block=False)
            plt.savefig(f'runs_resnet50_nograd/ROCs/{detector}_{attack}.png')

    print('done')
    pass
