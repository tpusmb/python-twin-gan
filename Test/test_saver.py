#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import timeit
import logging.handlers

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/test_saver.log",
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

if __name__ == "__main__":
    import tensorflow as tf

    v1 = tf.placeholder(tf.float32, name="v1")
    v2 = tf.placeholder(tf.float32, name="v2")
    v3 = tf.multiply(v1, v2)
    vx = tf.Variable(10.0, name="vx")
    v4 = tf.add(v3, vx, name="v4")
    saver = tf.train.Saver([vx])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(vx.assign(tf.add(vx, vx)))
    result = sess.run(v4, feed_dict={v1: 12.0, v2: 3.3})
    print(result)
    saver.save(sess, "./model_ex1")
