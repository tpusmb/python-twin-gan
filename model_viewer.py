#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Save the translation model to the tensorboard

Usage:
    model_viewer.py <model-name>

Options:
    -h --help           Show this screen.
    <model-name>        Name of the model to launch a visualisation you can put
                        'cat' to see the human to cat model
                        'anime' to see the human to anime model

"""


from __future__ import absolute_import
import os
import tensorflow as tf
from docopt import docopt
import logging.handlers

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/pb_viwer.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

# Absolute path to the folder location of this python file
FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
TRANSFORM_TASK = {"cat": [os.path.join(FOLDER_ABSOLUTE_PATH, "human_to_cat_128", "128"), 128],
                  "anime": [os.path.join(FOLDER_ABSOLUTE_PATH, "twingan_256", "256"), 256]}
LOG_DIR = '/logs/tests/1/'


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

    arguments = docopt(__doc__)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=session_config)
    with sess.as_default():

        # Load model
        model_folder = TRANSFORM_TASK[arguments["<model-name>"].lower().strip()][0]
        model_exp = os.path.expanduser(model_folder)
        PYTHON_LOGGER.info('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filename(model_exp)
        PYTHON_LOGGER.info('Metagraph file: %s' % meta_file)
        PYTHON_LOGGER.info('Checkpoint file: %s' % ckpt_file)

        # Restore the tensorflow session
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    # Make the write the current graph for tensorboard
    train_writer = tf.summary.FileWriter(LOG_DIR)
    train_writer.add_graph(sess.graph)
    PYTHON_LOGGER.info("Now you can look the graph by launching this command: tensorboard --logdir=/logs/tests/1/")
