import cv2
import pandas as pd
import pywt
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from model import Model
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load('output/weight.pth', map_location=device))
model.eval()


def normalize(x):
    maximum = x.max()
    minimum = x.min()
    if maximum == minimum:
        return x
    return (x - minimum)/(maximum - minimum)


if __name__ == '__main__':
    folder = 'data/crack-identification-ce784a-2020-iitk/test/'
    df = pd.read_csv('data/crack-identification-ce784a-2020-iitk/sample-submission.csv')
    classes = list()
    for index, row in tqdm(df.iterrows()):
        image_address = folder + row['filename']

        image = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(444, 444))
        coeffs2 = pywt.dwt2(image, 'bior1.3')
        _, (LH, HL, HH) = coeffs2
        LH = normalize(F.to_tensor(LH))
        HL = normalize(F.to_tensor(HL))
        HH = normalize(F.to_tensor(HH))

        image = torch.cat([LH, HL, HH], dim=0)

        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            output = model(image.float().unsqueeze(0).to(device))
        if torch.argmax(output, dim=1) == 0:
            predict = 'uncracked'
        else:
            predict = 'cracked'
        df.loc[index, 'class'] = predict

    df.to_csv('data/crack-identification-ce784a-2020-iitk/submission_3.csv', index=False)
    print()