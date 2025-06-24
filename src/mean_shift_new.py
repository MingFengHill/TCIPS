import torch
import numpy as np
from torch import sqrt, exp
import time
import math


class MeanShift_GPU():
    ''' Do meanshift clustering with GPU support'''
    def __init__(self,bandwidth = 2.5, batch_size = 1000, max_iter = 10, eps = 1e-5, check_converge = False):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.eps = eps # use for check converge
        self.cluster_eps = 1e-1 # use for check cluster
        self.check_converge = check_converge # Check converge will take 1.5 time longer
          
    def distance_batch(self,a,B):
        ''' Return distance between each element in a to each element in B'''
        return sqrt(((a[None,:] - B[:,None])**2)).sum(2)
    
    def distance(self,a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def fit(self,data):
        with torch.no_grad():
            n = len(data)
            if not data.is_cuda:
                data_gpu = data.cuda()
                X = data_gpu.clone()
            else:
                X = data.clone()
            #X = torch.from_numpy(np.copy(data)).cuda()
            
            for _ in range(self.max_iter):
                max_dis = 0;
                for i in range(0,n,self.batch_size):
                    s = slice(i,min(n,i+ self.batch_size))
                    if self.check_converge:
                        dis = self.distance_batch(X,X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch;
                        weight = dis
                        weight = self.gaussian(dis, self.bandwidth)
                    else:
                        weight = self.gaussian(self.distance_batch(X,X[s]), self.bandwidth)
                    num = (weight[:,:,None]*X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:,None]                    
                    
                #import pdb; pdb.set_trace()
                #Check converge
                if self.check_converge:
                    if max_dis < self.eps:
                        print("Converged")
                        break
            
            # end_time = time.time()
            # print("algorithm time (s)", end_time- begin_time)
            # Get center and labels
            if True:
                # Convert to numpy cpu show better performance
                points = X.cpu().data.numpy()
                labels, centers = self.cluster_points(points)
            else:
                # use GPU
                labels, centers = self.cluster_points(points)
                
            labels = np.array(labels)
            centers = np.array(centers)
            return labels,centers,X
        
    def gaussian(self,dist,bandwidth):
        return exp(-0.5*((dist/bandwidth))**2)/(bandwidth*math.sqrt(2*math.pi))
        
    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for j,center in enumerate(cluster_centers):
                    dist = self.distance(point, center)
                    if(dist < self.cluster_eps):
                        cluster_ids.append(j)
                        break
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids, cluster_centers


def mean_shift_gpu(x, bandwidth=1.31, batch_size=700):
    ms = MeanShift_GPU(bandwidth = bandwidth, batch_size = batch_size, max_iter = 10, eps = 1e-5, check_converge = False)
    labels, centers, X_fea = ms.fit(x)
    return labels

