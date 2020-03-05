#bgjb93209uromf.xzcjlcjlh82t3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class PCA():
    def __init__(self):
        self.preprocessing()

    def preprocessing(self):
        im = Image.open('6.jpg')
        #im.show()
        pix = np.array(im)
        print(pix.shape)
        bw_pix = np.mean(pix,axis=-1)
        img = Image.fromarray(bw_pix)



if __name__ == '__main__':
    p = PCA()