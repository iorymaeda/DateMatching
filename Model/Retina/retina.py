import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#---------------------------

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
# This comes with deepface!
from retinaface.model import retinaface_model
from retinaface.commons import preprocess, postprocess

#---------------------------

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 2:
    import logging
    tf.get_logger().setLevel(logging.ERROR)

#---------------------------
from .types import Landmarks, Face


#---------------------------

class RetinaFace:
    def __init__(self, normalize=True):
        self.model = None
        self.normalize = normalize
        self.build_model()
    
    def __call__(self, img_path:str, target_size=(224, 224)) -> list[np.ndarray]:
        img = self.get_image(img_path)
        obj = self.detect_faces(img)
        imgs = self.extract_faces(img, obj)
        imgs = [self.postprocces(img, target_size=target_size) for img in imgs]
        return imgs
    
    
    def build_model(self):
        if self.model is None:
            self.model = tf.function(
                retinaface_model.build_model(),
                input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
            )
        return self.model
    
    
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
    
    
    def detect_faces(self, img_path:str|np.ndarray, threshold=0.9, allow_upscaling=True) -> list[Face]:
        img = self.get_image(img_path)
        
        #---------------------------

        if self.model is None:
            self.build_model()

        #---------------------------

        nms_threshold = 0.4; decay4=0.5

        _feat_stride_fpn = [32, 16, 8]

        _anchors_fpn = {
            'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
            'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
            'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
        }

        _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

        #---------------------------

        proposals_list = []
        scores_list = []
        landmarks_list = []
        im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
        net_out = self.model(im_tensor)
        net_out = [elt.numpy() for elt in net_out]
        sym_idx = 0

        for _idx, s in enumerate(_feat_stride_fpn):
            _key = 'stride%s'%s
            scores = net_out[sym_idx]
            scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = _num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = _anchors_fpn['stride%s'%s]
            anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_deltas = bbox_deltas
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
            proposals = postprocess.bbox_pred(anchors, bbox_deltas)

            proposals = postprocess.clip_boxes(proposals, im_info[:2])

            if s==4 and decay4<1.0:
                scores *= decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= im_scale
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3]//A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
            landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            landmarks[:, :, 0:2] /= im_scale
            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list)
        if proposals.shape[0]==0:
            landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

        #nms = cpu_nms_wrapper(nms_threshold)
        #keep = nms(pre_det)
        keep = postprocess.cpu_nms(pre_det, nms_threshold)

        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        landmarks = landmarks[keep]

        resp = []
        for idx, face in enumerate(det):
            _landmarks = Landmarks(
                right_eye=list(landmarks[idx][0]), 
                left_eye=list(landmarks[idx][1]),
                nose=list(landmarks[idx][2]),
                mouth_right=list(landmarks[idx][3]),
                mouth_left=list(landmarks[idx][4]))
            
            face = Face(
                score=face[4], 
                facial_area=list(face[0:4].astype(int)),
                landmarks=_landmarks)
            resp.append(face)
            
        return resp
    
    
    def extract_faces(self, img:np.ndarray|str,  faces_obj:list[Face]=None, align=True) -> list[np.ndarray]:
        assert isinstance(img, (np.ndarray, str))
        
        if faces_obj is None:
            faces_obj = self.detect_faces(img)
        
        imgs = []
        for identity in faces_obj:
            if identity:
                facial_area = identity["facial_area"]
                facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

                if align == True:
                    landmarks = identity["landmarks"]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    nose = landmarks["nose"]
                    mouth_right = landmarks["mouth_right"]
                    mouth_left = landmarks["mouth_left"]

                    facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye, nose)

                imgs.append(facial_img[:, :, ::-1])
            
        return imgs
    

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