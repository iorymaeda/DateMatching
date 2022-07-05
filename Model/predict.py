"""This script load all photo in current dir, pocces and predict beauty score"""

import os
import pickle
import argparse
from tabulate import tabulate

import torch
from PIL import Image
from sklearn.decomposition import PCA
from facenet_pytorch import MTCNN, InceptionResnetV1

import utils

def load_photo() -> list:
    l = []
    img = None
    files = os.listdir()
    for file in files:
        if file != "tmp.png":
            if ".png" in file:
                img = Image.open(file)
                l.append([img.convert('RGB'), file])

            elif ".jpg"in file:
                img = Image.open(file)
                l.append([img, file])

    return l



if __name__ == "__main__":
    mtcnn = MTCNN()
    feauture_generator_model = InceptionResnetV1(pretrained='vggface2').eval()
    
    checkpoint = torch.load('Train/weights/w/prod.torch', map_location=torch.device('cpu'))
    classification_model = utils.Model(32, [64, 64]).eval()
    classification_model.load_state_dict(checkpoint['model'])

    with open("Train/weights/pca.pkl", 'rb') as f:
        pca: PCA = pickle.load(f)


    imgs = load_photo()
    files = [img[1] for img in imgs]
    if not imgs:
        print("File not founnd")
        exit()

    with torch.no_grad():
        imgs_cropped: list[torch.Tensor] = []

        for img in imgs:
            cropped = mtcnn(img[0], save_path='tmp.png')
            if cropped is not None: 
                imgs_cropped.append(cropped[None, ...])

        if len(imgs_cropped) > 1:
            imgs_cropped: torch.Tensor = torch.cat(imgs_cropped)

        elif len(imgs_cropped) == 1:
            imgs_cropped: torch.Tensor = imgs_cropped[0]

        # Check for founded face
        if len(imgs_cropped) >= 1:
            embs = feauture_generator_model(imgs_cropped)

            
            if len(imgs_cropped) >= 2:
                corr = []
                for emb1, f1 in zip(embs, files):
                    _corr = []
                    for emb2, f2 in zip(embs, files):
                        _corr.append(float(emb1@emb2.T))

                    corr.append(_corr)

                print(tabulate(corr, headers=files))

            transformed = pca.transform(embs).astype('float32')
            score: torch.Tensor = classification_model(torch.from_numpy(transformed))
            scores = score.sigmoid()
            print(scores)
            with open('score.txt', 'w') as f:
                for score in scores:
                    f.write(str(float(score)) + "\n")