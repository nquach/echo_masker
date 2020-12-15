import os
import pydicom as dcm
import numpy as np 
from utils import *
from model_zoo import *
from skimage.morphology import remove_small_objects
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from scipy.ndimage.morphology import binary_fill_holes
import skimage.transform as trans

#GLOBALS
NROWS = 384
NCOLS = 384

def active_contour(mask, niter, smooth=3, checkerboard_init=True):
	if checkerboard_init:
		init_ls = checkerboard_level_set(mask.shape, 6)
	else:
		init_ls = (mask > THRESHOLD)
	mask = morphological_chan_vese(mask, niter, init_level_set=init_ls, smoothing=smooth)
	return mask

def postprocess(mask, use_active_contour=False):
	mask = np.squeeze(mask)
	mask = (mask > 0.5)
	if use_active_contour:
		mask = active_contour(mask, niter=10, smooth=1, checkerboard_init=False)
	mask = binary_fill_holes(mask)
	mask = remove_small_objects(mask, min_size=200)
	return mask

#input is a 2D numpy arr
#model is a Keras model that is already initialized and with weights loaded in.
def create_mask(input_img, model):
	#Resize input_img to (NROWS,NCOLS) 
	#Use np.reshape to add batch and channel dim (NROWS,NCOLS) -> (1, NROWS, NCOLS, 1)
	#Use model.predict to on your resized img input
	#Use worker function above postprocess, to postprocess your mask. Can experiment with active contours feature which may help smooth. may not be necessary

	#Resize the mask with your helper function resize_img() in utils. remove singleton dim before hand np.squeeze()!
	#return resized mask
	pass

if __name__ == '__main__':
	#Given a folder with dicoms, do the following:
	'''
	for each dicom file
	1) open dicom file, extract pixel array, get first img (or any other frame will do too)
	2) initialize model, read in saved weights. 
	3) predict binary mask using model, then postprocess raw mask. Resize final mask to the original frame size
	4) save this mask somewhere, and make sure it can be tied to which dicom it came from. as png or jpeg
	'''
	weights_path = ''
	model = unet(pretrained_weights = weights_path, input_size = input_size, lr=1e-5)
	pass


