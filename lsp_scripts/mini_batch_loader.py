#!/usr/bin/env python2.7
# coding:utf-8
 
import collections
import os
import numpy as np
import cv2
import cPickle
 
PathInfo = collections.namedtuple('PathInfo', ['image_path', 'label_path'])
 
 
class MiniBatchLoader(object):
    LABEL_COUNT = 9
 
    def __init__(self, image_dir_path, batch_size, in_size):
 
        # load data paths
        self.data_dir = image_dir_path
        self.batch_size = batch_size
 
        # load a mean image
        self.mean = np.array([103.939, 116.779, 123.68])
        self.in_size = in_size
 
    # test ok
    def load_data(self, lines):
        mini_batch_size = self.batch_size
        in_channels = 3
        xs = np.zeros((mini_batch_size, in_channels,  self.in_size, self.in_size)).astype(np.float32)
        ys = np.zeros((mini_batch_size, 9, self.in_size, self.in_size)).astype(np.float32)
 
        for i, line in enumerate(lines):
            datum = line.split(',')
            img_fn = '%s%s' % (self.data_dir, datum[0])
            
            #_/_/_/ xs: read image & joint _/_/_/  
            img = cv2.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in datum[1:]])         

            #_/_/_/ image cropping _/_/_/
            joints = joints.reshape((len(joints) / 2, 2))
            x, y, w, h = cv2.boundingRect(np.asarray([joints.tolist()]))

            pad_w_r = 1.7
            pad_h_r = 1.7
            x -= (w * pad_w_r - w) / 2
            y -= (h * pad_h_r - h) / 2
            w *= pad_w_r
            h *= pad_h_r

            x, y, w, h = [int(z) for z in [x, y, w, h]]
            x = np.clip(x, 0, img.shape[1] - 1)
            y = np.clip(y, 0, img.shape[0] - 1)
            w = np.clip(w, 1, img.shape[1] - (x + 1))
            h = np.clip(h, 1, img.shape[0] - (y + 1))
            img = img[y:y + h, x:x + w]   

            joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
            joints = joints.flatten()

            #_/_/_/ resize _/_/_/
            orig_h, orig_w, _ = img.shape
            joints[0::2] = joints[0::2] / float(orig_w) * 224
            joints[1::2] = joints[1::2] / float(orig_h) * 224
            img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_NEAREST)

            xs[i, :, :, :] = ((img - self.mean)/255).transpose(2, 0, 1)

            #_/_/_/ heatmap _/_/_/
            ksize = 27
            h, w, c = img.shape
            joints = joints.reshape((len(joints) / 2, 2))
            heatmap = np.zeros((9,h,w))
            gaussian = self.gauss2D((ksize,ksize), 1.5)*4
            for j in range(len(joints)):
                x = joints[j,0]-13
                xp = joints[j,0]+14
                y = joints[j,1]-13
                yp = joints[j,1]+14
                l = np.clip(x, 0, img.shape[0])
                r = np.clip(xp, 0, img.shape[0])
                u = np.clip(y, 0, img.shape[1])
                d = np.clip(yp, 0, img.shape[1])

                clipped = gaussian[u-y:ksize-(yp-d), l-x:ksize-(xp-r)]
                heatmap[j,u:d,l:r] = clipped

            ys[i, :, :, :] = heatmap
            
        return xs, ys

    def gauss2D(self,shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

