import os

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from deepface.detectors import MtcnnWrapper



class MTCNN:
    def __init__(self, normalize=False):
        self.model = MtcnnWrapper.build_model()
        self.normalize = normalize

    def __call__(self, img_path:str, target_size=(224, 224)) -> list[np.ndarray]:
        img = self.get_image(img_path)
        imgs = MtcnnWrapper.detect_face(mtcnn, img)
        imgs = [self.postprocces(img[0][..., ::-1], target_size=target_size) for img in imgs]
        return imgs


    def get_image(self, img_path:str|np.ndarray) -> np.ndarray:
        if type(img_path) == str:  # Load from file path
            if not os.path.isfile(img_path):
                raise ValueError(f"Input image file path ({img_path}) does not exist.")

            # ----------------------------------------
            # PIL is suck and detection is not working
            # img = Image.open(img_path)
            # img = img.convert('RGB')
            # # to BGR
            # img = np.array(img)[..., ::-1]
            # ----------------------------------------

            img = cv2.imread(img_path)

        elif isinstance(img_path, np.ndarray):  # Use given NumPy array
            img = img_path.copy()

        else:
            raise ValueError(f"Invalid image input. Only file paths or a NumPy array accepted. Got {type(img_path)}")

        # Validate image shape
        if len(img.shape) != 3 or np.prod(img.shape) == 0:
            raise ValueError("Input image needs to have 3 channels at must not be empty.")

        return img


    def postprocces(self, img: np.ndarray|str, target_size=(224, 224), grayscale=False):
        #img might be path, base64 or numpy array. Convert it to numpy whatever it is.
        img = self.get_image(img)
        base_img = img.copy()
        #--------------------------

        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")

        #--------------------------

        #post-processing
        if grayscale == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #---------------------------------------------------
        #resize image to expected shape

        # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            if grayscale == False:
                # Put the base image in the middle of the padded image
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
            else:
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

        #------------------------------------------

        #double check: if target image is not still the same size with target.
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)

        #---------------------------------------------------

        #normalizing the image pixels
        if self.normalize:
            img = img.astype('float32')
            #normalize input in [0, 1]
            img /= 255 

        #---------------------------------------------------
        return img

mtcnn = MtcnnWrapper.build_model()
