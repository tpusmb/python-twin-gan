#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import base64
import io

import cv2
import os
import sys
import logging.handlers

import numpy as np
import pika
from PIL import Image

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/android_simulation.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


def string_to_cv2_image(base64_string):
    """
    Take a base64 string and convert to opencv image
    :param base64_string: (string) base 64 string
    :return: (ndarray) opencv image
    """
    img_data = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_image_to_string(cv2_image):
    """
    Converte cv2 image to a base 64 string
    :param cv2_image: (ndarray) cv2 image
    :return: (string) base 64 string
    """
    # 1 - Transform the image to byte
    _, buffer = cv2.imencode('.jpg', cv2_image)
    # 2 - convert to byte 64 and trasform to string remove the b' symbol
    return str(base64.b64encode(buffer))[2:]


connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.1.3'))
channel = connection.channel()

channel.exchange_declare(exchange='task',
                         exchange_type='direct')

result = channel.queue_declare(exclusive=True)
queue_name = result.method.queue

channel.queue_bind(exchange='task',
                   queue=queue_name,
                   routing_key="result")


def callback(ch, method, properties, body):
    print(" [x] %r:%r" % (method.routing_key, body))
    cv2_image = string_to_cv2_image(body)
    cv2.imwrite("out.jpg", cv2_image)
    sys.exit()


channel.basic_consume(callback,
                      queue=queue_name,
                      no_ack=True)

img = cv2.imread("../demo/testmoi.png")
message = cv2_image_to_string(img)

channel.basic_publish(exchange='task',
                      routing_key="cat",
                      body=message)

channel.start_consuming()
connection.close()
