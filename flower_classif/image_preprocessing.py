import keras
import numpy as np
# for image processing
import cv2
import tarfile
import math

# we will crop and resize input images to IMG_SIZE x IMG_SIZE
IMG_SIZE = 250


def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """
    cropx = max(0, img.shape[0] - img.shape[1])
    cropy = max(0, img.shape[1] - img.shape[0])
    cropped_img = img[cropx//2:img.shape[0]-math.ceil(cropx/2),
                      cropy//2:img.shape[1]-math.ceil(cropy/2),
                      :]
    return cropped_img


def prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=True):
    # decode image raw bytes to matrix
    img = decode_image_from_raw_bytes(raw_bytes)
    # take squared center crop
    img = image_center_crop(img)
    # resize for our model
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if normalize_for_model:
        # prepare for normalization
        img = img.astype("float32")
        # normalize for model
        img = keras.applications.inception_v3.preprocess_input(img)
    return img


# reads bytes directly from tar by filename
# (slow, but ok for testing, takes ~6 sec)
def read_raw_from_tar(tar_fn, fn):
    with tarfile.open(tar_fn) as f:
        m = f.getmember(fn)
        return f.extractfile(m).read()
