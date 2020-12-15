import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
from utils import *
from model_zoo import *
import datetime

def trainGenerator(batch_size,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
					mask_color_mode = "grayscale", num_class = 2, target_size = (384,384), seed = 1):
	'''
	can generate image and mask at the same time
	use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
	if you want to visualize the results of generator, set save_to_dir = "your path"
	'''
	image_datagen = ImageDataGenerator(**aug_dict)
	mask_datagen = ImageDataGenerator(**aug_dict)
	image_generator = image_datagen.flow_from_directory(
		image_folder,
		class_mode = None,
		color_mode = image_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		seed = seed)
	mask_generator = mask_datagen.flow_from_directory(
		mask_folder,
		class_mode = None,
		color_mode = mask_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		seed = seed)
	train_generator = zip(image_generator, mask_generator)
	for (img,mask) in train_generator:
		if np.amax(img) > 1:
			img = img / 255
		yield (img,mask)

if __name__ == '__main__':
	root_direc = ''  
	img_direc = os.path.join(root_direc, 'images')
	label_direc = os.path.join(root_direc, 'labels')
	ckpt_direc = ''
	plot_direc = ''

	model_name = 'unet' + str(datetime.date.today()) + '.hdf5'
	ckpt_path = os.path.join(ckpt_direc, model_name)

	data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.0,
                    zoom_range=0.05,
                    fill_mode='nearest')

	batch_size = 2

	img_size = (384,384)

	input_size = (384,384,1)

	lr = 1e-5

	num_train_img = num_img_in_direc(os.path.join(train_direc, img_direc))

	steps_per_epoch = num_train_img//batch_size
	print('Steps per epoch: ', steps_per_epoch)

	num_epochs = 3000

	gen = trainGenerator(batch_size, img_direc, label_direc, data_gen_args, target_size=img_size, seed=1)

	model = unet(pretrained_weights = None, input_size = input_size, lr=lr)

	model_checkpoint = ModelCheckpoint(ckpt_path, monitor='loss',verbose=1, save_best_only=True)
	history = model.fit_generator(generator=gen,steps_per_epoch=steps_per_epoch, epochs=num_epochs,callbacks=[model_checkpoint])
	
	#Optional: visualize loss/acc/mIOU curves
	visualize_loss(history, model_name[:-5], plot_direc)

