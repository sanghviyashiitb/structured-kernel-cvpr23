from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv


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