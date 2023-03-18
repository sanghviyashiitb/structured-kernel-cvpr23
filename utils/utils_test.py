from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

def add_inset(im, kernel):
	if np.ndim(im) == 2:
		H, W = np.shape(im)
		H1, W1 = np.int32(H*0.2), np.int32(W*0.2)
		kernel = kernel/np.max(np.ravel(kernel))
		k1 = np.asarray(Image.fromarray(kernel).resize((W1,H1)), dtype=np.float32)
		im[H-H1:H, 0:W1] = k1
	if np.ndim(im) == 3:
		H, W, _  = np.shape(im)
		H1, W1 = np.int32(H*0.2), np.int32(W*0.2)
		kernel = kernel*255/np.max(np.ravel(kernel))
		k1 = np.asarray(Image.fromarray(kernel).resize((W1,H1)), dtype=np.float32)
		for idx in range(3):
			im[H-H1:H, 0:W1, idx] = k1
	return im


def img_register(im_true, im_est):
	img1_color = im_true # Image to be aligned
	img2_color = im_est  # Reference image to which img1_color is aligned

	img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
	img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
	height, width = img2.shape

	# Create ORB detector with 5000 features.
	orb_detector = cv.ORB_create(5000)

	# Find keypoints and descriptors.
	# The first arg is the image, second arg is the mask
	#  (which is not reqiured in this case).
	kp1, d1 = orb_detector.detectAndCompute(img1, None)
	kp2, d2 = orb_detector.detectAndCompute(img2, None)

	# Match features between the two images.
	# We create a Brute Force matcher with
	# Hamming distance as measurement mode.
	matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)


	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)

	# Sort matches on the basis of their Hamming distance.
	matches.sort(key = lambda x: x.distance)

	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches)*100)]
	no_of_matches = len(matches)

	# Define empty matrices of shape no_of_matches * 2.
	p1 = np.zeros((no_of_matches, 2))
	p2 = np.zeros((no_of_matches, 2))

	for i in range(len(matches)):
		p1[i, :] = kp1[matches[i].queryIdx].pt
		p2[i, :] = kp2[matches[i].trainIdx].pt

	# Find the homography matrix.
	homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

	# Use this matrix to transform the
	# colored image wrt the reference image.
	transformed_img = cv.warpPerspective(img1_color, homography, (width, height))

	return transformed_img, img2_color


def change_whitebalance(img_src, img_target):        
    mean_src = np.mean(img_src, axis=(0,1))
    mean_target = np.mean(img_target, axis=(0,1))

    for idx in range(3):
        img_src[:,:,idx] = img_src[:,:,idx].astype(np.float32)*mean_target[idx]/mean_src[idx]
    return img_src, img_target



def bayer_to_rggb(x_bayer):
	H, W = np.shape(x_bayer)
	x_rggb = []	

	x_rggb.append(x_bayer[0::2,0::2])
	x_rggb.append(x_bayer[0::2,1::2])
	x_rggb.append(x_bayer[1::2,0::2])
	x_rggb.append(x_bayer[1::2,1::2])
	
	return x_rggb



def rggb_to_rgb(x_list, switch = False, coeff=[1,1,1]):
	H, W = np.shape(x_list[0])
	x_rgb = np.zeros([H, W, 3])
	
	x_rgb[:,:,0] = x_list[0]
	x_rgb[:,:,1] = (x_list[1]+x_list[2])*0.5
	x_rgb[:,:,2] = x_list[3]

	if switch:
		x_rgb = np.flip(x_rgb,2)

	x_rgb[:,:,0] *= coeff[0]
	x_rgb[:,:,1] *= coeff[1]
	x_rgb[:,:,2] *= coeff[2]

	return x_rgb