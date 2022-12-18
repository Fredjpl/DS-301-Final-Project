import os, glob, inspect, tqdm
import numpy as np
import cv2
import tensorflow as tf
# import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from skimage.transform import resize

#PACK_PATH = os.path.dirname("../Data/")
PACK_PATH = os.path.join("C:/Users/35132/Desktop/machine learning/gan/")
print(PACK_PATH)
class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize

        self.x_tot, self.x_tr, self.x_te = [], [], []
        print(os.path.join(PACK_PATH, "xinggan_face_512_clear", "*.jpg"))
        paths_png = self.sorted_list(os.path.join(PACK_PATH, "xinggan_face_512_clear", "*.jpg"))[:50000]
        # for idx_p, path_png in enumerate(tqdm.tqdm(paths_png)):
        #     img_origin = plt.imread(path_png)
        #     [h, w, c] = img_origin.shape
        #     self.x_tot.append(img_origin)
        self.x_tot = paths_png
        print(1)
        bound = int(len(self.x_tot) * 0.9)
        print(bound)
        self.x_tr = self.x_tot[:bound]
        self.x_te = self.x_tot[bound:]

        self.x_tr, self.x_te = np.asarray(self.x_tr), np.asarray(self.x_te)

        # self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        # self.x_te = np.ndarray.astype(self.x_te, np.float32)

        self.num_tr, self.num_te = len(self.x_tr), len(self.x_te)
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample = cv2.imread(self.x_te[0])

        self.height = x_sample.shape[0]//4
        self.width = x_sample.shape[1]//4
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        self.num_class = 2

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
        print("Class  %d" %(self.num_class))
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

    def read_data(self, datalist):
        data_tmp= []

        for idx_p, path_png in enumerate(datalist):
            img_origin = cv2.imread(path_png)
            img_origin = cv2.resize(img_origin, (self.height, self.width))
            data_tmp.append(img_origin)

        return np.array(data_tmp)

    def sorted_list(self, path):

        tmplist = glob.glob(path)
        tmplist.sort()

        return tmplist

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr = self.x_tr[start:end]

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr = shuffle(self.x_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        x_tr = self.read_data(x_tr)
        if(x_tr.shape[0] != batch_size):
            x_tr = x_tr[-1-batch_size:-1]

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)


        return x_tr, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size
        x_te = self.x_te[start:end]

        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end
        x_te = self.read_data(x_te)

        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        return x_te, terminator

if __name__ == "__main__":
    dataset = Dataset()