#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import logging.handlers

import numpy as np
import tensorflow as tf
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

        :param model_path: (string) Absolute path to the model to load
        :param image_hw:(int) Height and width of the output image
        :param input_tensor_name: Name of the input layer for tensorflow
        :param output_tensor_name: Name of the output layer for tensorflow
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

            self.load_model(model_path, input_map=input_map)

            # Get input and output tensors
            self.output = tf.get_default_graph().get_tensor_by_name(output_tensor_name)

    def translate(self, input_image):
        """
        Translate the input image
        :param input_image: (ndarray) opencv image to translate
        :return: (ndarray) opencv image translated
        """
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        feed_dict = {self.images_placeholder: rgb_image}
        output = self.sess.run(self.output, feed_dict=feed_dict)
        output = output[0] * 255.0  # Batch size == 1, range = 0~1.
        output = output.astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr

    # Adapted from https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
    def load_model(self, model, input_map=None):
        """
        Loads a tensorflow model and restore the variables to the default session.
        :param model: (string) absolute path to the model
        :param input_map: (Dictionary) empty input for the network
        """
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if os.path.isfile(model_exp):
            PYTHON_LOGGER.info('Model filename: %s' % model_exp)
            with tf.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=input_map, name='')
        else:
            PYTHON_LOGGER.info('Model directory: %s' % model_exp)
            meta_file, ckpt_file = self.get_model_filename(model_exp)

            PYTHON_LOGGER.info('Metagraph file: %s' % meta_file)
            PYTHON_LOGGER.info('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    @staticmethod
    def get_model_filename(model_dir):
        """
        Get the model file name
        :param model_dir: (String) Absolute path to the model
        :return: (string) Model name
        """
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            meta_file = ckpt_file + '.meta'
            return meta_file, ckpt_file
        else:
            raise ValueError('No checkpoint file found in the model directory (%s)' % model_dir)


if __name__ == "__main__":

    # Read a image
    img1 = cv2.imread("demo/testmoi.png")
    img2 = cv2.imread("demo/testmoi2.png")
    img1 = cv2.resize(img1, (600, 600))
    img2 = cv2.resize(img2, (600, 600))
    tow_images = np.concatenate((img1, img2), axis=0)
    cv2.imshow("main", tow_images)
    cv2.waitKey(0)
    image_translation = ImageTranslation("human_to_cat_128/128", 128)
    PYTHON_LOGGER.info("Start translation")
    # Translate to a cat image
    output_image = image_translation.translate(img1)
    cv2.imshow("main", output_image)
    cv2.waitKey(0)
    output_image = image_translation.translate(img2)
    cv2.imshow("main", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
