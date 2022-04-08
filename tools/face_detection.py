import sys
import numpy as np
from PIL import Image
from dlib import cnn_face_detection_model_v1 as face_detect_model

cnn_face_detector = None

def shrink(img, max_length=2048):
        ow, oh = img.size
        if max_length >= max(ow, oh):
            return img, 1.0

        if ow > oh:
            mult = max_length / ow
        else:
            mult = max_length / oh
        w = int(ow * mult)
        h = int(oh * mult)
        return img.resize((w, h), Image.BILINEAR), mult


def face_detection(
        img_path,
        verbose=False,
        model_file='tools/dlib_face_detector/mmod_human_face_detector.dat'):
    """
    Detects faces using dlib cnn face detection, and extend the bounding box
    to include the entire face.
    """

    global cnn_face_detector
    if cnn_face_detector is None:
        cnn_face_detector = face_detect_model(model_file)

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img_shrinked, mult = shrink(img)

    im = np.asarray(img_shrinked)
    if len(im.shape) != 3 or im.shape[2] != 3:
        return []

    crop_ims = []
    dets = cnn_face_detector(im, 0)
    for k, d in enumerate(dets):
        top = d.rect.top() / mult
        bottom = d.rect.bottom() / mult
        left = d.rect.left() / mult
        right = d.rect.right() / mult

        wid = right - left
        left = max(0, left - wid // 2.5)
        top = max(0, top - wid // 1.5)
        right = min(w - 1, right + wid // 2.5)
        bottom = min(h - 1, bottom + wid // 2.5)

        if d.confidence > 1:
            if verbose:
                print("%d-th face detected: (%d, %d, %d, %d)" %
                      (k, left, top, right, bottom))
            crop_im = img.crop((left, top, right, bottom))
            crop_ims.append((crop_im, (left, top, right, bottom)))

    return crop_ims


def detect_face(img_path):
    faces = face_detection(img_path, verbose=False)
    if len(faces) == 0:
        print("no face detected by dlib")
        print(img_path)
        # sys.exit()
        img = Image.open(img_path).convert('RGB')
        box = img.getbbox()
    else:
        img, box = faces[0]

    return img, box
