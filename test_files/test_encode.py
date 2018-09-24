#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import base64

import cv2
import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
import logging.handlers

import numpy as np

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/test_encode.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# 1 - Transforme the image to byte
_, buffer = cv2.imencode('.jpg', cv2.imread("test.jpg"))
# 2 - Converte to byte 64 and trasform to string remove the b' symbol
png_as_text = str(base64.b64encode(buffer))[2:]
# 3 - Converte the byte string to image
img = stringToImage(png_as_text.encode())
# 4 - Converte to opencv format BGR
img = toRGB(img)
cv2.imshow("main", img)
cv2.waitKey(0)
