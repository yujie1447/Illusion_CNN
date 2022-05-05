#!/usr/bin/env python
# coding: utf-8

# In[10]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from pylab import *
import os, shutil, os.path as path
from skimage import filters
import numpy as np
print(matplotlib.__version__)

# # Helper function

# In[11]:


def init_colors(color1, color2):
    if color1 is None:
        color1 = [1,1,1]
    elif color1 == 'random':
        color1 = np.random.rand(3)
    if color2 is None:
        color2 = [0,0,0]
    elif color2 == 'random':
        color2 = np.random.rand(3)
    return color1, color2

def grating(size=[100,100], ori=0, bar_width=5, phase=0, waveform='sin',
        color1=None, color2=None, contrast=1, shape=[224,224], alpha=True):
    '''
    Generate a grating of specified orientation, spatial frequency, phase, etc.
    '''
    color1, color2 = init_colors(color1, color2)
    H, W = shape
    x, y = np.meshgrid(np.arange(W)-(W-1)/2, np.arange(H)[::-1]-(H-1)/2)
    h, w = size
    outline = ((x/(w/2))**2 + (y/(w/2))**2 < 1)[...,np.newaxis]
    texture = 0.5 + 0.5*contrast*np.cos(np.pi/bar_width * (x*np.cos(ori)+y*np.sin(ori)) + phase)[...,np.newaxis] 
    if waveform == 'square':
        texture = (texture > 0.5)
    texture = texture * np.reshape(color1, [1,1,-1]) + (1-texture) * np.reshape(color2, [1,1,-1])
    if alpha:
        im = np.dstack([texture, outline])
    else:
        im = outline * texture + (1-outline) * 0.5
    return im

def band_limit_noise(sigma=5, color1=None, color2=None, contrast=1, shape=[224,224], alpha=True):
    '''
    Generate a noisy background image (band-limited Gaussian random noise).
    '''
    color1, color2 = init_colors(color1, color2)
    im = np.random.rand(*shape)
    im = filters.difference_of_gaussians(im, low_sigma=sigma, high_sigma=None, mode='wrap')
    im = 0.5 + 0.5*contrast*im/np.max(np.abs(im))
    im = im[...,np.newaxis]
    im = im * np.reshape(color1, [1,1,-1]) + (1-im) * np.reshape(color2, [1,1,-1])
    if alpha:
        im = np.dstack([im, np.ones(im.shape[:2])])
    return im

def alpha_blend(im1, im2, alpha='max'):
    '''
    Overlay im1 on im2, blending according to the alpha channel of im1.
    '''
    im = im1[...,3:] * im1[...,:3] + (1-im1[...,3:]) * im2[...,:3]
    if alpha is None:
        alpha = np.ones(im.shape[:2])
    elif alpha == 'max':
        alpha = np.maximum(im1[...,3], im2[...,3])
    im = np.dstack([im, alpha])
    return im


# # Simple gratings without inducer

# In[12]:


 

def generate_simple_grating(foldername, ori_inner):
    print(foldername)
    os.makedirs(foldername)
    k = 0
    for ori in ori_inner:
        for phase in phases:
            for contrast in contrasts:
                for n in range(N):
                    inner = grating(size=grating_size, ori=-ori, bar_width=bar_width, phase=phase, contrast=contrast)
                    bg = band_limit_noise(color2=None)
                    im = alpha_blend(inner, bg)
                    imsave(f"{foldername}/{k:04d}_ori{ori/np.pi*180:.0f}_w{bar_width:02d}_p{phase*180/np.pi:.0f}_c{contrast*100:03d}.jpg", im)
                    k += 1
bar_width = 10
grating_size = [70,70]
phases = np.linspace(0, np.pi, 5) # np.arange(0, 2*np.pi, 2*np.pi/5)
contrasts = [1]
N = 20
stages = ['train', 'val']
ori_inners = np.arange(-np.pi/4, 0, 5/180*np.pi)
for stage in stages:
    shutil.rmtree(f'/Users/wuyujie/Documents/2020NMA/gratings/{stage}')
    generate_simple_grating(f'/Users/wuyujie/Documents/2020NMA/gratings/{stage}/0-left', ori_inners)
    generate_simple_grating(f'/Users/wuyujie/Documents/2020NMA/gratings/{stage}/1-right', -ori_inners)
    generate_simple_grating(f'/Users/wuyujie/Documents/2020NMA/gratings/{stage}/2-vertical', [0])


# # illusory gratings with inducer

# In[ ]:




def generate_illusory_grating(foldername, ori_inners, ori_outers):
    print(foldername)
    os.makedirs(foldername)
    k = 0
    for ori_inner in ori_inners:
        for ori_outer in ori_outers:
            for phase in phases:
                for contrast in contrasts:
                    for n in range(N):
                        inner = grating(size=grating_size_inner, ori=-ori_inner, bar_width=bar_width, phase=phase, contrast=contrast)
                        outer = grating(size=grating_size_outer, ori=-ori_outer, bar_width=bar_width, phase=phase, contrast=contrast)
                        bg = band_limit_noise(color2=None)
                        im = alpha_blend(inner, outer)
                        im = alpha_blend(im, bg)
                        imsave(f"{foldername}/{k:04d}_orii{ori_inner/np.pi*180:.0f}_orio{ori_outer/np.pi*180:.0f}_w{bar_width:02d}_p{phase*180/np.pi:.0f}_c{contrast*100:03d}.jpg", im)
                        k += 1
bar_width = 10
grating_size_inner = [70]*2
grating_size_outer = [180]*2
phases = np.linspace(0, np.pi, 5)  # np.arange(0, 2*np.pi, 2*np.pi/5)
contrasts = [1]
N = 2
stages = ['test']
ori_inners = [0]
ori_outers = np.arange(-np.pi/6, 0, 5/180*np.pi)

for stage in stages:
    if path.exists(f'/Users/wuyujie/Documents/2020NMA/test_gratings/{stage}'):
        shutil.rmtree(f'/Users/wuyujie/Documents/2020NMA/test_gratings/{stage}')
    generate_illusory_grating(f'/Users/wuyujie/Documents/2020NMA/test_gratings/{stage}/0-left', ori_inners, -ori_outers)
    generate_illusory_grating(f'/Users/wuyujie/Documents/2020NMA/test_gratings/{stage}/1-right', ori_inners, ori_outers)
    generate_illusory_grating(f'/Users/wuyujie/Documents/2020NMA/test_gratings/{stage}/2-vertical', ori_inners, [0])

