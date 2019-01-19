"""
``Reconstructing vehicles from a single image`` demo

This code illustrates a heavily simplified, efficient, and robust pipeline abridged from
the approach outlined in the ICRA 2017 paper "Reconstructing vehicles from a single image:
Shape priors for road-scene understanding".

This demo code uses 'cars' as the objects of interest (as this was the motivating example
from the paper), nevertheless works across many object categories that shape priors can be
defined over.

Before running this script, we assume that an object detector has detected bounding boxes in
the image, and that we have semantic keypoints detected by a keypoint detection network (we use
the stacked-hourglass architecture, in our work).
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import numpy as np
import os
from skimage import io
import subprocess
import sys

import utils


if __name__ == '__main__':

	# Create a 'cache' directory if it does not already exist
	if not os.path.exists('cache'):
		os.makedirs('cache')
		print('Created dir: ', 'cache')

	"""
	Filepaths
	"""
	poseAdjusterInput = os.path.join('cache', 'poseAdjusterInput.txt')
	poseAdjusterOutput = os.path.join('cache', 'poseAdjusterOutput.txt')
	poseAdjusterOutputAux = os.path.join('cache', 'shapeAfterPose.txt')
	shapeAdjusterInput = os.path.join('cache', 'shapeAdjusterInput.txt')
	shapeAdjusterOutput = os.path.join('cache', 'shapeAdjusterOutput.txt')
	shapeAdjusterOutputAux = os.path.join('cache', 'lambdasAfterShape.txt')
	poseAdjusterCommand = os.path.join('ceresCode', 'build', 'singleViewPoseAdjuster')
	shapeAdjusterCommand = os.path.join('ceresCode', 'build', 'singleViewShapeAdjuster')


	"""
	Load/Define KITTI-specific parameters
	"""

	# Camera intrinsics
	K = np.loadtxt('data/intrinsics.txt')

	# Height above the ground (in meters) at which the camera is mounted on the car
	CAM_HEIGHT = 1.72


	"""
	Load/Define shape-prior-related parameters
	"""

	# Number of keypoints
	NUM_KEYPOINTS = 36

	# Average car dimensions (in meters)
	AVG_CAR_HEIGHT = 1.5208
	AVG_CAR_WIDTH = 1.6362
	AVG_CAR_LENGTH = 3.8600

	# Shape prior (mean shape, basis vectors)
	meanShape = np.loadtxt('data/meanShape.txt')
	basisVectors = np.loadtxt('data/basisVectors.txt')
	numBasisVectors = basisVectors.shape[0]

	# Initialize the shape coefficients
	shapeCoefficients = np.loadtxt('data/shapeCoefficientsInitial.txt')
	numShapeCoefficients = shapeCoefficients.shape[0]

	# Keypoint-visibility lookup table (some prior of what keypoints should be visible from
	# a particular viewpoint)
	keypointVisibility = np.loadtxt('data/keypointLookupAzimuth.txt')


	"""
	Visualization-specific definitions
	"""

	# Edges (keypoints that are to be connected, for visualizing a car wireframe)
	edges = [(i,i+1) for i in range(2,17)] + [(i,i+1) for i in range(20,35)]
	edges += [(17,2), (35,20)]
	edges += [(i,i+18) for i in range(2,17)]
	# # Template code to draw wireframe lines in 2D and 3D (used while plotting)
	# lines2D = [[(x[i,0], x[i,1]), (x[j,0], x[j,1])] for (i,j) in edges]
	# lines3D = [[(xs[i], ys[i], zs[i]), (xs[j], ys[j], zs[j])] for (i,j) in edges]
	# lc2D = LineCollection(lines)
	# lc3D = Line3DCollection(lines)


	"""
	Read in a sample image, its bounding box detection(s), and keypoint information
	"""

	exampleID = '01'
	exampleDir =  os.path.join('data', 'examples', exampleID)
	# Image
	img = io.imread(os.path.join(exampleDir, 'image.png'))
	# Car bounding box
	bbox = np.loadtxt(os.path.join(exampleDir, 'bbox.txt'))
	# Keypoint predictions from a stacked-hourglass network
	# (Note: these predictions are made by resizing the car bounding box image to 64 x 64,
	# and passing it through the stacked-hourglass network.)
	keypoints = np.loadtxt(os.path.join(exampleDir, 'keypoints.txt'))
	keypoints = keypoints.reshape((NUM_KEYPOINTS, 3))
	# We ignore the last column, which contains CNN confidence scores
	keypoints = keypoints[:,0:2]
	# Map keypoints from 64 x 64 to the actual image pixel coordinates (use the bbox for this)
	keypoints[:,0] = (keypoints[:,0] * bbox[2]) / 64
	keypoints[:,1] = (keypoints[:,1] * bbox[3]) / 64
	keypoints[:,0] = keypoints[:,0] + bbox[0] - 1
	keypoints[:,1] = keypoints[:,1] + bbox[1] - 1
	# Read in a viewpoint estimate of the azimuth (yaw) of the car. In our work, we use a CNN
	# that predicts object viewpoints to initialize this variable. The approach works best if
	# the initial azimuth is within +/- 30 degrees of the true azimuth. There are several
	# off-the-shelf networks one could use for this task. We tried with the viewpoint network from
	# the CVPR 2015 paper "Viewpoints and Keypoints" and with the one from the "RenderForCNN"
	# paper at ICCV 2015.
	azimuth = np.loadtxt(os.path.join(exampleDir, 'azimuth.txt'))


	"""
	Initialize a wireframe model that represents the shape of the car, using a guess of the pose
	obtained from monocular cues
	"""

	# Compute (rough) dimensions (length, width, height) of the car shape prior
	l = np.abs(np.max(meanShape[:,0]) - np.min(meanShape[:,0]))
	h = np.abs(np.max(meanShape[:,1]) - np.min(meanShape[:,1]))
	w = np.abs(np.max(meanShape[:,2]) - np.min(meanShape[:,2]))

	# Compute (anisotropic) scaling factors for each axis so that the dimensions of
	# the car shape prior become roughly equal to those of an average KITTI car.
	# The canonical wireframe is defined such that the width of the car is aligned with
	# the Z-axis, the height with the Y-axis, and the length with the X-axis.
	sz = AVG_CAR_WIDTH / w
	sy = AVG_CAR_HEIGHT / h
	sx = AVG_CAR_LENGTH / l

	# Scale the mean shape and the basis vectors
	meanShape_scaled = np.copy(meanShape)
	meanShape_scaled[:,0] = sx * meanShape[:,0]
	meanShape_scaled[:,1] = sy * meanShape[:,1]
	meanShape_scaled[:,2] = sz * meanShape[:,2]
	basisVectors_scaled = np.copy(basisVectors)
	for b in range(numBasisVectors):
		basisVectors_scaled[b,0::3] = sx * basisVectors_scaled[b,0::3]
		basisVectors_scaled[b,1::3] = sy * basisVectors_scaled[b,1::3]
		basisVectors_scaled[b,2::3] = sz * basisVectors_scaled[b,2::3]

	# Rotate the "scaled" meanShape and basisVectors from the shape prior coordinate frame
	# to the KITTI coordinate frame.
	R = utils.rotZ(180, 'degrees').dot(utils.rotY(90, 'degrees'))
	# After rotating, we also need to apply the azimuth (yaw) guess
	R = utils.rotY(azimuth + 1.5*np.pi).dot(R)

	meanShape_scaled_rotated = np.copy(meanShape_scaled)
	for n in range(NUM_KEYPOINTS):
		meanShape_scaled_rotated[n,:] = R.dot(meanShape_scaled_rotated[n,:].reshape((3,1))).T
	basisVectors_scaled_rotated = np.copy(basisVectors_scaled)
	for b in range(numBasisVectors):
		for n in range(NUM_KEYPOINTS):
			basisVectors_scaled_rotated[b,3*n:3*(n+1)] = \
			R.dot(basisVectors_scaled_rotated[b,3*n:3*(n+1)].reshape((3,1))).T

	# Using the height above the ground at which the camera is mounted on the car, we compute an
	# approximate location of the wirefreme in 3D (in meters!!).
	# Assuming that the bottom of the car bbox lies on the road, we can back project the bbox
	# bottom (mid-point of the bottom of the bbox), via the ground plane, to 3D.
	bboxBottomInImage = np.asarray([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]-1, 1]).T
	n = np.asarray([0, -1., 0])
	numerator = np.linalg.solve(K, bboxBottomInImage)
	denominator = n.dot(numerator)
	bboxBottomIn3D = np.reshape(-CAM_HEIGHT * (numerator/denominator) + \
		np.asarray([0, -AVG_CAR_HEIGHT/2, 0]), (3,1))

	# Incorporate a translational offset that factors in the azimuth estimate
	# Currently, the 3D location of the bbox bottom in 3D does not take into account the
	# orientation of the vehicle. Here's where that gets accounted for.
	carBottomFaceRect = np.asarray([[-AVG_CAR_LENGTH/2, -AVG_CAR_WIDTH/2], \
		[AVG_CAR_LENGTH/2, -AVG_CAR_WIDTH/2], [AVG_CAR_LENGTH/2, AVG_CAR_WIDTH/2],
		[-AVG_CAR_LENGTH/2, AVG_CAR_WIDTH/2]])
	rot2D = np.matrix([[np.cos(azimuth), -np.sin(azimuth)], [np.sin(azimuth), np.cos(azimuth)]])
	carBottomFaceRect_rotated = np.copy(carBottomFaceRect)
	for b in range(carBottomFaceRect.shape[0]):
		b2D = carBottomFaceRect[b,:].reshape((2,1))
		b2D = rot2D.dot(b2D)
		carBottomFaceRect_rotated[b,:] = b2D.reshape((2))
	idxMin = np.unravel_index(np.argmin(carBottomFaceRect_rotated), carBottomFaceRect_rotated.shape)
	idxMax = np.unravel_index(np.argmax(carBottomFaceRect_rotated), carBottomFaceRect_rotated.shape)
	offset = abs(carBottomFaceRect_rotated[idxMax] - carBottomFaceRect_rotated[idxMin]) / 2
	bboxBottomIn3D[2,0] += offset

	# Translate the scaled and rotated mean shape
	# Note: the basis vectors need not be translated, because they're defined centerd about
	# the mean shape.
	meanShape_scaled_rotated_translated = np.copy(meanShape_scaled_rotated)
	meanShape_scaled_rotated_translated += np.tile(bboxBottomIn3D, (1,NUM_KEYPOINTS)).T


	"""
	Compute the initialized wireframe in 2D (for plotting purposes)
	"""
	wireframe2D_init = np.zeros((3,NUM_KEYPOINTS))
	wireframe2D_init[2,:] = 1.
	for k in range(NUM_KEYPOINTS):
		x2D = K.dot(meanShape_scaled_rotated_translated[k,:].reshape((3,1)))
		x2D = x2D / x2D[2,0]
		wireframe2D_init[:,k] = x2D.reshape((3))

	"""
	Write input file for pose adjuster
	"""

	inFile = open(poseAdjusterInput, 'w')
	# Number of keypoints
	inFile.write(str(NUM_KEYPOINTS) + '\n')
	# 3D location guess of the center of the bottom of the car
	inFile.write(str(bboxBottomIn3D[0].item()) + ' ' + str(bboxBottomIn3D[1].item()) + ' ' + \
		str(bboxBottomIn3D[2].item()) + '\n')
	# Average dimensions
	inFile.write(str(AVG_CAR_HEIGHT) + ' ' + str(AVG_CAR_WIDTH) + ' ' + str(AVG_CAR_LENGTH) \
		+ '\n')
	# Camera intrinsics
	inFile.write(str(K[0][0]) + ' ' + str(K[0][1]) + ' ' + str(K[0][2]) + \
		' ' + str(K[1][0]) + ' ' + str(K[1][1]) + ' ' + str(K[1][2]) + \
		' ' + str(K[2][0]) + ' ' + str(K[2][1]) + ' ' + str(K[2][2]) + '\n')
	# 2D keypoint detections
	for k in range(NUM_KEYPOINTS):
		inFile.write(str(keypoints[k,0]) + ' ' + str(keypoints[k,1]) + '\n')
	# Viewpoint-prior weights (from lookup table)
	if azimuth >= 0:
		thetaDeg = int(np.ceil(np.rad2deg(azimuth)))
	else:
		thetaDeg = int(np.ceil(np.rad2deg(azimuth + 2*np.pi)))
	viewpointWeights = keypointVisibility[thetaDeg,:]
	viewpointWeights = 0.75 * viewpointWeights + 0.1 * np.full(viewpointWeights.shape, 1.)
	for i in range(viewpointWeights.shape[0]):
		inFile.write(str(viewpointWeights[i]) + '\n')
	# Mean shape
	for i in range(meanShape_scaled_rotated_translated.shape[0]):
		cur = meanShape_scaled_rotated_translated[i,:]
		inFile.write(str(cur[0]) + ' ' + str(cur[1]) +' ' + str(cur[2]) + '\n')
	# Basis vectors
	inFile.write(str(numBasisVectors) + '\n')
	for i in range(basisVectors_scaled_rotated.shape[0]):
		for j in range(basisVectors_scaled_rotated.shape[1]):
			inFile.write(str(basisVectors_scaled_rotated[i,j]) + ' ')
		inFile.write('\n')
	# Shape coefficients
	for i in range(len(shapeCoefficients)):
		inFile.write(str(shapeCoefficients[i]) + ' ')
	inFile.write('\n')
	inFile.close()

	"""
	Perform pose adjustment
	"""
	print('Performing pose adjustment ...')
	subprocess.call('./' + poseAdjusterCommand)

	"""
	Read pose adjuster output
	"""
	predictedPose = np.loadtxt(poseAdjusterOutput)
	wireframe3D_afterPoseAdjustment = np.loadtxt(poseAdjusterOutputAux)
	wireframe2D_afterPoseAdjustment = np.zeros((3,NUM_KEYPOINTS))
	wireframe2D_afterPoseAdjustment[2,:] = 1.
	for k in range(NUM_KEYPOINTS):
		x2D = K.dot(wireframe3D_afterPoseAdjustment[k,:].reshape((3,1)))
		x2D = x2D / x2D[2,0]
		wireframe2D_afterPoseAdjustment[:,k] = x2D.reshape((3))


	"""
	Write shape adjuster input file
	"""

	inFile = open(shapeAdjusterInput, 'w')
	# Number of keypoints
	inFile.write(str(NUM_KEYPOINTS) + '\n')
	# 3D location guess of the center of the bottom of the car
	inFile.write(str(bboxBottomIn3D[0].item()) + ' ' + str(bboxBottomIn3D[1].item()) + ' ' + \
		str(bboxBottomIn3D[2].item()) + '\n')
	# Average dimensions
	inFile.write(str(AVG_CAR_HEIGHT) + ' ' + str(AVG_CAR_WIDTH) + ' ' + str(AVG_CAR_LENGTH) \
		+ '\n')
	# Camera intrinsics
	inFile.write(str(K[0][0]) + ' ' + str(K[0][1]) + ' ' + str(K[0][2]) + \
		' ' + str(K[1][0]) + ' ' + str(K[1][1]) + ' ' + str(K[1][2]) + \
		' ' + str(K[2][0]) + ' ' + str(K[2][1]) + ' ' + str(K[2][2]) + '\n')
	# 2D keypoint detections
	for k in range(NUM_KEYPOINTS):
		inFile.write(str(keypoints[k,0]) + ' ' + str(keypoints[k,1]) + '\n')
	# Viewpoint-prior weights (from lookup table)
	if azimuth >= 0:
		thetaDeg = int(np.ceil(np.rad2deg(azimuth)))
	else:
		thetaDeg = int(np.ceil(np.rad2deg(azimuth + 2*np.pi)))
	viewpointWeights = keypointVisibility[thetaDeg,:]
	viewpointWeights = 0.75 * viewpointWeights + 0.1 * np.full(viewpointWeights.shape, 1.)
	for i in range(viewpointWeights.shape[0]):
		inFile.write(str(viewpointWeights[i]) + '\n')
	# Mean shape (Note that we are writing in the initialized 3D wireframe and NOT the
	# pose adjusted wireframe. The Ceres code takes care of the conversion. Writing the
	# pose adjusted wireframe will likely mess up the results)
	for i in range(meanShape_scaled_rotated_translated.shape[0]):
		cur = meanShape_scaled_rotated_translated[i,:]
		inFile.write(str(cur[0]) + ' ' + str(cur[1]) +' ' + str(cur[2]) + '\n')
	# Basis vectors
	inFile.write(str(numBasisVectors) + '\n')
	for i in range(basisVectors_scaled_rotated.shape[0]):
		for j in range(basisVectors_scaled_rotated.shape[1]):
			inFile.write(str(basisVectors_scaled_rotated[i,j]) + ' ')
		inFile.write('\n')
	# Shape coefficients
	for i in range(len(shapeCoefficients)):
		inFile.write(str(shapeCoefficients[i]) + ' ')
	inFile.write('\n')
	for i in range(9):
		inFile.write(str(predictedPose[i]) + ' ')
	inFile.write('\n')
	for i in range(9,12):
		inFile.write(str(predictedPose[i]) + ' ')
	inFile.write('\n')
	inFile.close()

	"""
	Perform shape adjustment
	"""
	print('Performing shape adjustment ...')
	subprocess.call('./' + shapeAdjusterCommand)

	"""
	Read shape adjuster output
	"""
	predictedShape = np.loadtxt(shapeAdjusterOutputAux)
	wireframe3D_afterShapeAdjustment = np.loadtxt(shapeAdjusterOutput)
	wireframe2D_afterShapeAdjustment = np.zeros((3,NUM_KEYPOINTS))
	wireframe2D_afterShapeAdjustment[2,:] = 1.
	for k in range(NUM_KEYPOINTS):
		x2D = K.dot(wireframe3D_afterShapeAdjustment[k,:].reshape((3,1)))
		x2D = x2D / x2D[2,0]
		wireframe2D_afterShapeAdjustment[:,k] = x2D.reshape((3))


	"""
	Visualization
	"""

	fig, ax = plt.subplots(1)
	ax.imshow(img)
	# Plot keypoints
	ax.scatter(keypoints[:,0], keypoints[:,1], c='r')
	rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', \
		facecolor='none')
	ax.add_patch(rect)
	w2D = wireframe2D_init
	lines2D = [[(w2D[0,i], w2D[1,i]), (w2D[0,j], w2D[1,j])] for (i,j) in edges]
	lc2D = LineCollection(lines2D)
	ax.add_collection(lc2D)
	plt.savefig('cache/01.png')
	plt.close()

	fig, ax = plt.subplots(1)
	ax.imshow(img)
	# Plot keypoints
	ax.scatter(keypoints[:,0], keypoints[:,1], c='r')
	rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', \
		facecolor='none')
	ax.add_patch(rect)
	w2D = wireframe2D_afterPoseAdjustment
	lines2D = [[(w2D[0,i], w2D[1,i]), (w2D[0,j], w2D[1,j])] for (i,j) in edges]
	lc2D = LineCollection(lines2D)
	ax.add_collection(lc2D)
	plt.savefig('cache/02.png')
	plt.close()

	fig, ax = plt.subplots(1)
	ax.imshow(img)
	# Plot keypoints
	ax.scatter(keypoints[:,0], keypoints[:,1], c='r')
	rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', \
		facecolor='none')
	ax.add_patch(rect)
	w2D = wireframe2D_afterShapeAdjustment
	lines2D = [[(w2D[0,i], w2D[1,i]), (w2D[0,j], w2D[1,j])] for (i,j) in edges]
	lc2D = LineCollection(lines2D)
	ax.add_collection(lc2D)
	plt.savefig('cache/03.png')
	plt.close()
