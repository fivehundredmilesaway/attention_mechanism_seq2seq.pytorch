import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import numpy as np 

class sampling_schedule(nn.Module):
    def __init__(self,final_iter=200000,threshold=0.6,sampling_method="Linear_decay"):
        '''
        now_iter: the current num of iter
        final_iter: after final_iter the prob of sampling will not continue to decrease
        threshold:the lowest prob
        '''
        self.final_iter=final_iter
        self.threshold=threshold
        self.sampling_method=sampling_method
        super(sampling_schedule,self).__init__()

    def sampling(self, target, y, now_iter,sampling_method=None):
        if self.sampling_method is None:
            return target
        elif self.sampling_method == "Linear_decay":
            k = 1
            c = (k - self.threshold) / self.final_iter
            sampling_prob = max(self.threshold, k - c * now_iter)
        elif sampling_method == "Exponential_decay":
            x=math.exp(math.log(self.threshold)/self.final_iter)
            sampling_prob=pow(x,now_iter)
        else:
            raise ValueError("Unsupport sampling method")
        candidate = torch.stack((target, y))
        batch_size=y.size()
        sampling_list = np.random.choice(a=2, size=batch_size, p=[sampling_prob, 1 - sampling_prob])
        sampling_list = torch.LongTensor(sampling_list).unsqueeze(0).to(y.device)
        y = torch.gather(candidate, dim=0, index=sampling_list).squeeze(0)
        return y
