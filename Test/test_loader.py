#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import timeit
import logging.handlers

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/test_loader.log",
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
LOG_DIR = '/logs/tests/1/'

if __name__ == "__main__":
    import tensorflow as tf
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./model_ex1.meta")
        saver.restore(sess, "./model_ex1")
        result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
        print(result)

    train_writer = tf.summary.FileWriter(LOG_DIR)
    train_writer.add_graph(sess.graph)
