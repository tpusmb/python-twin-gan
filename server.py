#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the main server to get images from the android app and translate to cat, anime or other

Usage:
    server.py <ip>

Options:
    -h --help           Show this screen.
    <ip>                Ip adress of the rabbitmq server

"""

from __future__ import absolute_import
import base64
import cv2
import io
import os
import logging.handlers
import numpy as np
import pika
from PIL import Image
from image_translation import ImageTranslation
from docopt import docopt
import threading

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/server.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

TRANSFORM_TASK = {"cat": ["human_to_cat_128/128", 128], "anime": ["twingan_256/256", 256]}


class Worker(threading.Thread):

    def __init__(self, server_ip, routing_key, translation_algorithm_model_path, image_width_height):
        """

        :param server_ip:
        :param routing_key:
        :param translation_algorithm_model_path:
        :param image_width_height:
        """
        super().__init__()
        self.translation_algorithm_model_path = translation_algorithm_model_path

        PYTHON_LOGGER.info("Start rabbit mq connection with the worker {}".format(self.translation_algorithm_model_path))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=server_ip))
        self.channel = self.connection.channel()

        self.channel.exchange_declare(exchange='task',
                                      exchange_type='direct')

        result = self.channel.queue_declare(exclusive=True)
        queue_name = result.method.queue
        self.channel.queue_bind(exchange='task',
                                queue=queue_name,
                                routing_key=routing_key)

        self.channel.basic_consume(self.call_back,
                                   queue=queue_name,
                                   no_ack=True)

        self.algorithm_translation = ImageTranslation(translation_algorithm_model_path, image_width_height)

    @staticmethod
    def string_to_cv2_image(base64_string):
        """
        Take a base64 string and convert to opencv image
        :param base64_string: (string) base 64 string
        :return: (ndarray) opencv image
        """
        img_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
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

    def call_back(self, ch, method, _, body):

        PYTHON_LOGGER.info("Get message with key {}".format(method.routing_key))
        cv2_image = self.string_to_cv2_image(body)

        try:
            output_image = self.algorithm_translation.translate(cv2_image)
        except Exception as e:
            PYTHON_LOGGER.error("Error in the image translation: {}".format(e))
            output_image = None

        if output_image is None:
            body_string = "none"
        else:
            body_string = self.cv2_image_to_string(output_image)

        PYTHON_LOGGER.info("Send the output image")
        self.channel.basic_publish(exchange='task',
                                   routing_key="result",
                                   body=body_string)

    def run(self):
        """

        :return:
        """
        PYTHON_LOGGER.info("Start the thread {}".format(self.translation_algorithm_model_path))
        self.channel.start_consuming()


class Server:

    def __init__(self, ip):
        """
        Connect to a rabbit mq
        :param ip: (string) ip of the rabbit mq server
        """
        PYTHON_LOGGER.info("Start rabbit mq connection")

        thread_list = list()

        for transform_name in TRANSFORM_TASK:
            model_path, img_width_height = TRANSFORM_TASK[transform_name]
            thread_list.append(Worker(ip, transform_name, model_path, img_width_height))

        PYTHON_LOGGER.info("Start all algorithm")
        for worker in thread_list:
            worker.run()

        PYTHON_LOGGER.info("Wait the end of algorithm")
        for worker in thread_list:
            worker.join()

    def start(self):
        """
        Start lisening
        :return:
        """


    def stop(self):
        """
        Close the server
        :return:
        """
        PYTHON_LOGGER.info("Stop the server")
        self.connection.close()


if __name__ == "__main__":

    arguments = docopt(__doc__)
    server = Server(arguments["<ip>"])
    server.start()
    server.stop()
