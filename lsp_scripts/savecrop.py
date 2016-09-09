import collections
import os
import numpy as np
import cv2
import cPickle

test_fn = "data/FLIC-small/test_joints18.csv"
test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
data_dir = "data/FLIC-small/images/"

for i, line in enumerate(test_dl):
            datum = line.split(',')
            img_fn = '%s%s' % (data_dir, datum[0])
            
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
            cv2.imwrite('testcrop/'+str(i)+'.jpg',img)
