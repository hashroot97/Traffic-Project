import tensorflow as tf
import os
from PIL import Image
import matplotlib

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([128, 128], Image.ANTIALIAS)
    return image

def main():
    ## Source Image Directory
	image_dir = 'C:\\Users\\aksha\\Downloads\\Traffic_Project\\Data\\Main_empty'
	len_image_dir = len(os.listdir(image_dir))
	print('Found %d images' %len_image_dir)
    #Destination Image Directory
	resized_dir = 'C:\\Users\\aksha\\Downloads\\Traffic_Project\\Data\\Res_Main_Empty'
	for im in os.listdir(image_dir):
		with open(image_dir+'\\'+im, 'r+b') as f:
			image = Image.open(f).convert('RGB')
			image = resize_image(image)
			image.save(os.path.join(resized_dir, im), image.format)
	print('Done')

if __name__ == '__main__':
	main()
