"""This script load all photo in current dir, pocces and predict beauty score"""

print('load libraries')

import io
import sys
import pickle
import pathlib
import argparse
from tabulate import tabulate

import cv2
import torch
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from deepface import DeepFace
from sklearn.decomposition import PCA


import utils

DETECTOR_BACKEND = 'mtcnn'


def load_faces(save=True) -> list:
    imgs = []
    current_path = pathlib.Path()
    for file in current_path.iterdir():
        if file.is_file() and file.suffix in ['.png', '.jpg', '.jpeg'] and file.stem != "heatmap":
            _imgs = detector(str(file), (embeder.input_shape_x, embeder.input_shape_y))
            for idx, img in enumerate(_imgs):
                idx = "" if idx == 0 else f" {idx}"
                face_name = 'tmp/'+file.stem + idx + file.suffix

                Image.fromarray(img).save(face_name)
                imgs.append([img, face_name])
    return imgs


def heatmap_similarity(images_emb: np.ndarray, imgs: np.ndarray, robust=False):
    fig, ax = plt.subplots(1, 1, figsize=(2*len(imgs), 2*len(imgs)))
    fig.patch.set_facecolor('white')
    
    corr = images_emb @ images_emb.T * 100
    mask = np.zeros_like(corr)
    mask[range(mask.shape[0]), range(mask.shape[0])] = True

    ax = sns.heatmap(
        corr,
        vmin=0 if robust else None,
        vmax=100 if robust else None,
        mask=mask if robust else None,
        annot=True,
        cmap="RdBu",
        square=True,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
    )

    for text in ax.texts:
        text.set_size(20)
        if text.get_text() == '1e+02':
            text.set_text('100')

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    heatmap = PIL.Image.open(buf)
    plt.close(fig)

    square_size = int(heatmap.size[0] / len(images_emb))
    arr = np.zeros(shape = (heatmap.size[0]+square_size, heatmap.size[0]+square_size, 3), dtype=np.uint8)
    arr[:, :, :] = 255
    heatmap_with_img = PIL.Image.fromarray(arr, mode = 'RGB')
    heatmap_with_img.paste(heatmap, (square_size, square_size))

    pos = 0
    for idx, img in enumerate(imgs):
        pos+= square_size
        img = cv2.resize(img, (square_size, square_size))
        heatmap_with_img.paste(PIL.Image.fromarray(img), (pos, 0))

    pos = 0
    for idx, img in enumerate(imgs):
        pos+= square_size
        img = cv2.resize(img, (square_size, square_size))
        heatmap_with_img.paste(PIL.Image.fromarray(img), (0, pos))
        
    return heatmap_with_img


class Embeder:
    """This class just call predict function from model"""
    def __init__(self, model_name:str):
        self._model_name = model_name
        self._model = DeepFace.build_model(model_name)
        self.input_shape_x, self.input_shape_y = DeepFace.functions.find_input_shape(self._model)
        
        
    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)
        
        img = self.normalize_input(img)
        if len(img.shape) == 3:
            img = img[None, ...]

        elif len(img.shape) != 4:
            raise Exception(f'Something wrong with image shape: {img.shape}')
            
        print([i.mean() for i in img])
        pred = self._model.predict(img)
        pred = self.normilize_output(pred)
        return pred


    def normilize_output(self, x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)


    def normalize_input(self, img):
        if img.max() > 1:
            return img / 255
        else:
            return img


if __name__ == "__main__":
    

    with tf.device('/cpu:0'):
        if DETECTOR_BACKEND == 'mtcnn':
            from MTCNN import MTCNN
            detector = MTCNN(normalize=False)

        elif DETECTOR_BACKEND == 'retinaface':
            from Retina import RetinaFace
            detector = RetinaFace(normalize=False)

        else:
            raise Exception('Invalid DETECTOR_BACKEND')

        embeder = Embeder('Facenet512')
        checkpoint = torch.load('Train/weights/w/prod.torch', map_location=torch.device('cpu'))
        classification_model = utils.Model(32, [64, 64]).eval()
        classification_model.load_state_dict(checkpoint['model'])
        with open("Train/weights/pca.pkl", 'rb') as f:
            pca: PCA = pickle.load(f)

        print('Load faces')
        imgs = load_faces()
        if not imgs:
            raise Exception('Files or faces not founnd')

        files = [img[1] for img in imgs]
        faces = [img[0] for img in imgs]

        if len(faces) > 1:
            faces = np.stack(faces)
        elif len(faces) == 0:
            faces = faces[0]
        else:
            raise Exception('Faces not founnd')
        
        print('Get face embeddings')
        embs: np.ndarray = embeder(faces.astype('float32', copy=False))
                
        if len(faces) >= 2:
            print('Plot heatmap')
            heatmap: Image = heatmap_similarity(embs, faces, robust=True)
            heatmap.save('heatmap.png')
            print(tabulate(embs @ embs.T, headers=files))

        transformed: np.ndarray = pca.transform(embs)
        transformed: np.ndarray = transformed.astype('float32')
        scores: torch.Tensor = classification_model(torch.from_numpy(transformed))
        scores: torch.Tensor = scores.sigmoid()
        print(scores)

        with open('score.txt', 'w') as f:
            for score in scores:
                f.write(str(float(score)) + "\n")