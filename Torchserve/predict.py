"""This script load all photo in current dir, pocces and predict beauty score"""

import os
import pickle
import argparse
from wsgiref import headers
from tabulate import tabulate

import torch
from PIL import Image
from sklearn.decomposition import PCA
from facenet_pytorch import MTCNN, InceptionResnetV1


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
    import os
    import sys
    import inspect

    # Import from parent directory
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 
    import utils


    mtcnn = MTCNN()
    feauture_generator_model = InceptionResnetV1(pretrained='vggface2').eval()
    
    checkpoint = torch.load('../Models/w/prod.torch')
    classification_model = utils.Model(32, [64, 64]).eval()
    classification_model.load_state_dict(checkpoint['model'])

    with open("../Models/pca.pkl", 'rb') as f:
        pca: PCA = pickle.load(f)


    imgs = load_photo()
    files = [img[1] for img in imgs]
    if not imgs:
        print("File not founnd")
        exit()

    with torch.no_grad():
        imgs_cropped: list[torch.Tensor] = [mtcnn(img[0], save_path='tmp.png')[None, ...] for img in imgs]
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