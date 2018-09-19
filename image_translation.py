#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import logging.handlers
import tensorflow as tf
import util_io
import cv2

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/image_translation.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


class ImageTranslation(object):
    def __init__(self, model_path, image_hw=256, input_tensor_name="sources_ph",
                 output_tensor_name="custom_generated_t_style_source:0"):
        """

        :param model_path:
        :param image_hw:
        :param input_tensor_name:
        :param output_tensor_name:
        """
        # Load the model
        PYTHON_LOGGER.info('Loading inference model')
        session_config = tf.ConfigProto(allow_soft_placement=True, )
        self.sess = tf.Session(config=session_config)
        with self.sess.as_default():
            input_map = None
            if input_tensor_name:
                self.images_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
                image = tf.image.convert_image_dtype(self.images_placeholder, dtype=tf.float32)
                image = tf.image.resize_images(image, (image_hw, image_hw))
                image = tf.expand_dims(image, axis=0)
                input_map = {input_tensor_name: image}

            util_io.load_model(model_path, input_map=input_map)

            # Get input and output tensors
            self.output = tf.get_default_graph().get_tensor_by_name(output_tensor_name)

    def translate(self, input_image):
        """
        Translate the input image
        :param input_image: (ndarray) image to translate
        :return:
        """
        # util_io.imread(image_path, dtype=np.uint8)
        feed_dict = {self.images_placeholder: input_image}
        output = self.sess.run(self.output, feed_dict=feed_dict)
        output = output[0] * 255.0  # Batch size == 1, range = 0~1.
        return output


if __name__ == "__main__":

    image_translation = ImageTranslation("human_to_cat_128/128", 128)
    output = image_translation.translate(cv2.imread("demo/testmoi.jpg"))
    pass
