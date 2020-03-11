

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class PCA():
    def __init__(self,percentage,rand=False,save=True,graph=False):
        self.im,self.bw_pix,self.centralized = self.preprocessing()
        self.eval, self.evec = self.Get_Eigenvalues()
        self.norms = []
        for p in percentage:
            self.norms.append(self.Reconstruct(p,rand,save))
        if graph:
            plt.plot(percentage,self.norms)
            #plt.semilogx(percentage,self.norms)
            plt.title('Reconstruction Error vs percentage of principal components')
            plt.xlabel('percentage of principal components')
            plt.ylabel('Reconstruction Error')
            plt.show()

    def preprocessing(self):
        im = Image.open('6.jpg')
        #im.show()
        pix = np.array(im)
        bw_pix = np.mean(pix,axis=-1)
        self.mean =  np.mean(bw_pix,axis=0)
        centralized = bw_pix - self.mean
        return im, bw_pix, centralized



    def Get_Eigenvalues(self):
        covar = np.cov(self.centralized)
        w,v = np.linalg.eig(covar)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]
        return w,v


    def Reconstruct(self,p,rand,save):
        r = ''
        if rand==True:
            np.random.shuffle(self.evec)
            r = 'rand' 
        vec = self.evec[:,0:int(p*len(self.eval))]


        fin_data = vec.T@self.centralized
        img_rec = vec@fin_data+self.mean
        F_N = np.linalg.norm(img_rec - self.bw_pix)

        if save:
            err_img = Image.fromarray(img_rec-self.bw_pix)
            err_img = err_img.convert('RGB')
            err_img.save("Error{}{}.jpg".format(r,int(p*100)))
        
            fi_img = Image.fromarray(img_rec)
            fi_img = fi_img.convert('RGB')
            fi_img.save("Final{}{}.jpg".format(r,int(p*100)))
        
            print(
                "Error with top {}% principal components : {}".format(100*p,F_N))
        return(F_N)




        




if __name__ == '__main__':
    p1 = PCA([0.1,0.25,0.5,1])
    p2 = PCA([0.1,0.25,0.5,1],rand=True)
    p3 = PCA(np.arange(0.05,1,0.01),rand=False,save=False,graph=True)
