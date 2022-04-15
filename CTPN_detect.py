# coding=utf-8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import glob
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
class detection:
    def __init__(self,car):
        self.car = car
        self.plate = None

    def app_path(self):
        """Returns the base application path."""
        if hasattr(sys, 'frozen'):
            # Handles PyInstaller
            return os.path.dirname(sys.executable)  # 使用pyinstaller打包后的exe目录
        return os.path.dirname(__file__)  # 没打包前的py目录

    def resize_image(self,img):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return re_im, (new_h / img_size[0], new_w / img_size[1])

    def text_detect(self,image, checkpoint):
        tf.reset_default_graph()
        with tf.get_default_graph().as_default():
            # 模型参数定义
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # 加载模型
                ckpt_state = tf.train.get_checkpoint_state(checkpoint)
                model_path = os.path.join(checkpoint, os.path.basename(ckpt_state.model_checkpoint_path))
                saver.restore(sess, model_path)
                # 预测文本框位置
                img = image

                # img=cv2.resize(img,(816,608))
                img, (rh, rw) = self.resize_image(img)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)
        return boxes


    def ctpn_detect(self):
        self.car = cv2.resize(self.car, (816, 608))
        self.file = self.app_path() + r'/ctpn_model/'
        self.is_file = os.path.exists(self.file)
        if self.is_file == False:
            return
        boxes = self.text_detect(self.car, self.file)
        x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, 0, 0, 0, 0, 0, 0
        max_area = 0
        for i, box in enumerate(boxes):
            x1, y1 = box[0], box[1]
            x3, y3 = box[4], box[5]
            wid = abs(x3 - x1)
            hei = abs(y3 - y1)
            area = wid * hei
            if area > max_area:
                max_area = area
                x2 = x1
                y2 = y1
                x4 = x3
                y4 = y3
        self.car = cv2.rectangle(self.car, (x2-5, y2-5), (x4+5, y4+5), (0, 0, 255), 2)
        if x1>0 and x2>0 and x3>0 and x4>0:
            self.plate=self.car[y2:y4,x2:x4]

