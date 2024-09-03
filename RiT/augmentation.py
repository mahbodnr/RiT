import numpy as np
import torch

class CutMix(object):
  def __init__(self, beta):
    self.beta = beta

  def __call__(self, batch):
    img, label = batch
    rand_img, rand_label = self._shuffle_minibatch(batch)
    lambda_ = np.random.beta(self.beta,self.beta)
    img_x, img_y = img.size(2), img.size(3)
    r_x = np.random.uniform(0, img_x)
    r_y = np.random.uniform(0, img_y)
    r_w = img_x * np.sqrt(1-lambda_)
    r_h = r_y * np.sqrt(1-lambda_)
    x1 = int(np.clip(r_x - r_w // 2, a_min=0, a_max=img_x))
    x2 = int(np.clip(r_x + r_w // 2, a_min=0, a_max=img_x))
    y1 = int(np.clip(r_y - r_h // 2, a_min=0, a_max=r_y))
    y2 = int(np.clip(r_y + r_h // 2, a_min=0, a_max=r_y))
    img[:, :, x1:x2, y1:y2] = rand_img[:, :, x1:x2, y1:y2]
    
    lambda_ = 1 - (x2-x1)*(y2-y1)/(img_x*img_y)
    return img, label, rand_label, lambda_

  def _shuffle_minibatch(self, batch):
    img, label = batch
    rand_img, rand_label = img.clone(), label.clone()
    rand_idx = torch.randperm(img.size(0))
    rand_img, rand_label = rand_img[rand_idx], rand_label[rand_idx]
    return rand_img, rand_label

# Code: https://github.com/facebookresearch/mixup-cifar10
class MixUp(object):
  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def __call__(self, batch):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x, y = batch
    lam = np.random.beta(self.alpha, self.alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
