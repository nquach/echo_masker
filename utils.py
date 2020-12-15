import os
import numpy as np
import pydicom as dcm
import pickle as pkl
import skimage.io as io  
import skimage.transform as trans 
import matplotlib.pyplot as plt
from tqdm import tqdm

def safe_makedir(path): 
	if not os.path.exists(path):
		os.makedirs(path)

def num_img_in_direc(path):
	#COMPLETE MEE
	#from absolute path of direc, counts number of image files, return this number
	pass

#Given a Numpy arr, resize to dimensions specified by tuple new_size. Returns resized array. Use skimage.transform.resize
#can assume img is 2D (no channel dim)
def resize_img(img_arr, new_size):  
	#COMPLETE ME
	pass

# imagefile = PyDicom Object
def dicom2imglist(imagefile):
	'''
	converts raw dicom to numpy arrays
	'''
	try:
		ds = imagefile
		nrow = int(ds.Rows)
		ncol = int(ds.Columns)
		ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
		img_list = []
		if len(ds.pixel_array.shape) == 4: #format is (nframes, nrow, ncol, 3)
			nframes = ds.pixel_array.shape[0]
			R = ds.pixel_array[:,:,:,0]
			B = ds.pixel_array[:,:,:,1]
			G = ds.pixel_array[:,:,:,2]
			gray = (0.2989 * R + 0.5870 * G + 0.1140 * B)
			for i in range(nframes):
				img_list.append(gray[i, :, :])
			return img_list
		elif len(ds.pixel_array.shape) == 3: #format (nframes, nrow, ncol) (ie in grayscale already)
			nframes = ds.pixel_array.shape[0]
			for i in range(nframes):
				img_list.append(np.invert(ds.pixel_array[i,:,:]))
			return img_list
	except:
		return None




def visualize_loss(history, model_name, direc):
	save_direc = os.path.join(direc, 'loss_plots')
	safe_makedir(save_direc)

	plt.figure()
	plt.plot(history.history['accuracy'])
	plt.title('Model Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	save_path = os.path.join(save_direc, model_name + 'accuracy.png')
	plt.savefig(save_path)
	plt.close()

	plt.figure()
	plt.plot(history.history['loss'])
	plt.title('Model Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	save_path = os.path.join(save_direc, model_name + 'loss.png')
	plt.savefig(save_path)
	plt.close()

	plt.figure()
	plt.plot(history.history['mean_iou'])
	plt.title('Model Mean IoU')
	plt.xlabel('Epoch')
	plt.ylabel('mIoU')
	save_path = os.path.join(save_direc, model_name + 'mIoU.png')
	plt.savefig(save_path)
	plt.close()