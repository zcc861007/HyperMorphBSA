"""
Coded by Donald Cantrell, MD, PhD (donald.cantrell@nm.org)
         Chaochao Zhou, PhD (chaochao.zhou@northwestern.edu / czhouphd@gmail.com)
Radiology Department, Northwestern University and Medicine, Chicago, IL
"""

from matplotlib import pyplot as plt
import IPython
import nibabel
import numpy as np
import random
import matplotlib
import math
import cv2
import os

def loadDSA(root_dir, folder):
    #image_file = image_file.decode('utf-8')
    # root_dir = '/data/ASA2019Nifti'
    # root_dir = data_dir
    input_img = nibabel.load(os.path.join(root_dir, folder, 'input.nii'))
    input_data = input_img.get_fdata()
    return input_data

def listdir_NoHidden_OnlyFolders(path):
    nohidden = [f for f in os.listdir(path) if not f.startswith('.')]
    nohidden_onlyFolders = [f for f in nohidden if os.path.isdir(os.path.join(path, f))]
    return nohidden_onlyFolders

def createFeaturePlot(frame1, frame2, printToScreen=True, maxGoodMatches = 50):
    # compute features -- keys and descriptors
    orb = cv2.ORB_create(edgeThreshold=10, patchSize=10, nlevels=20, fastThreshold=5,
                         scaleFactor=1.1, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=1000)
    keys1, des1 = orb.detectAndCompute(frame1, None)
    keys2, des2 = orb.detectAndCompute(frame2, None)
    # print(keys1)

    # compute matches
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_KDTREE = 0
    search_params = dict(checks=50)   # or pass empty dictionary
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = []
    matches = flann.knnMatch(des1, des2, k=2)

    matches = [i for i in matches if len(i)==2]

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x[1].distance)
    # print(matches)

    print(f'There are {len(matches)} matches.')

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(0,len(matches))]
    # print(matchesMask)
    # assert False

    # ratio test as per Lowe's paper
    numGoodMatches = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            numGoodMatches += 1
            matchesMask[i]=[1,0]
            if numGoodMatches > maxGoodMatches:
                break
    print(f'The number of good matches is {numGoodMatches}.')

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 2) # If this is zero, it will show all unmatched features

    img3 = cv2.drawMatchesKnn(frame1,keys1,frame2,keys2,matches,None,**draw_params)

    if printToScreen:
        plt.figure(figsize = (15,35))
        plt.imshow(img3,)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

    return img3

def getEightBitImage(angio_data, frame):
    img = angio_data[:,:,frame]
    scaled_img = img * (255.0 / 4095.0)
    img_uint8 = np.uint8(scaled_img)
    img_uint8 = np.repeat(img_uint8[:,:,np.newaxis], 3, axis=2)
    return img_uint8

def getKeysDescriptors(angio_data, frame):
    img = getEightBitImage(angio_data, frame)
    orb = cv2.ORB_create(edgeThreshold=10, patchSize=10, nlevels=20, fastThreshold=5,
                         scaleFactor=1.1, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=10000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def plotKeysOnImage(img, keypoints):
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, keypoints, np.array([]), color=(0,255,0), flags=0)
    plt.imshow(img2)
    plt.show()

def performDSA(img_matrix):
    img_matrix = img_matrix.astype(float)
    firstFrame = img_matrix[:,:,0]
    print(f'The shape of the first frame is {firstFrame.shape}.')
    dsa_matrix = img_matrix - firstFrame[:,:, np.newaxis]
    return dsa_matrix

def apply_warp(angio_data, warp_matrices, index):
    warp_matrix = warp_matrices[index]
    transformedFrame = angio_data[:,:,index]
    sz = angio_data[:,:,index].shape
    transformedFrame = cv2.warpAffine(transformedFrame, warp_matrix[0:2,0:3], (sz[1],sz[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0);
    #transformedFrame = cv2.warpAffine(transformedFrame, warp_matrix[0:2,0:3], (sz[1],sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0);
    return transformedFrame

def computeWarpMatrixWithFeatures(frame1, frame2):
    # compute features -- keys and descriptors
    orb = cv2.ORB_create(edgeThreshold=10, patchSize=10, nlevels=20, fastThreshold=5,
                         scaleFactor=1.1, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=10000)
    keys1, des1 = orb.detectAndCompute(frame1, None)
    keys2, des2 = orb.detectAndCompute(frame2, None)

    # compute matches
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_KDTREE = 0
    search_params = dict(checks=50)   # or pass empty dictionary
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matches = [i for i in matches if len(i)==2]
    print(f'There are {len(matches)} matches.')

    # refine to good matches
    GOOD_DISTANCE = 0.7
    good = []
    for m,n in matches:
        if m.distance < GOOD_DISTANCE*n.distance:
            good.append([m,n])

    # Extract location of good matches
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = keys1[match[0].queryIdx].pt
        points2[i, :] = keys2[match[0].trainIdx].pt

    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(points1, points2)

    return transformation_rigid_matrix

def getEightBitMatrix(angio_data):
    scaled_mat = (angio_data - np.amin(angio_data)) * (255.0 / np.amax(angio_data))
    mat_uint8 = np.uint8(scaled_mat)
    return mat_uint8

def getGrayScaleFrame(angio_data, frameNum):
    theFrame = angio_data[:,:,frameNum]
    img = np.repeat(theFrame[:,:,np.newaxis], 3, axis=2)
    return img

def generateAffineDSA(xrays):
    #Compute direct warps
    test_data = xrays.transpose(1,2,0).astype('float64')
    frames = test_data.shape[2]
    direct_matrices = []
    for frameNum in range(0, frames):
        print(f'I am on frame {frameNum}.')
        base_frame = getEightBitImage(test_data,0)
        aligned_frame = getEightBitImage(test_data, frameNum)
        #aligned_frame = np.float32(warp_to_index(test_data, warp_matrices, frameNum, 0))
        direct_matrix = computeWarpMatrixWithFeatures(base_frame, aligned_frame)
        print(direct_matrix)
        direct_matrices.append(direct_matrix)

    # Now let's try using the sequential warping to improve the dsa
    direct_moco_angio = np.zeros_like(test_data)
    for f in range(0,test_data.shape[2]):
        print(f'I am working on frame {f}')
        transformedFrame = apply_warp(test_data, direct_matrices, f)
        direct_moco_angio[:,:,f] = transformedFrame

    print(direct_moco_angio.shape, direct_moco_angio.dtype, direct_moco_angio.min(), direct_moco_angio.max())

    dsa_affine = performDSA(direct_moco_angio)
    dsa_affine = dsa_affine.transpose(2,0,1)
    dsa_affine = dsa_affine.astype('float32')
    # dsa_affine = (dsa_affine - dsa_affine.min()) / (dsa_affine.max() - dsa_affine.min()) 

    return dsa_affine   


if __name__ == '__main__':
    # workspace = 'drive/MyDrive/Colab Notebooks/MotionStabilize/'
    # data_dir = '/content/drive/MyDrive/Colab Notebooks/PixShift/PixShift_main/datasets/test'
    workspace = './'
    data_dir = '/data/ChaochaoData/ClearMatch/nifti_predictions/test_dataset'    

    if not os.path.exists(workspace + 'TestOutput'):
        os.makedirs(workspace + 'TestOutput')
        print(f'Created {workspace + "TestOutput"}')

    all_folders = listdir_NoHidden_OnlyFolders(data_dir)
    print(len(all_folders))
    print(all_folders)

    # nsel = np.random.choice(len(all_folders))
    nsel = 11
    print(nsel)
    test_data = loadDSA(data_dir, all_folders[nsel])
    print(test_data.shape, test_data.dtype, test_data.min(), test_data.max())

    dsa_original = performDSA(test_data)
    dsa_original = dsa_original.transpose(2,0,1)
    dsa_original = dsa_original.astype('float32')
    dsa_original = (dsa_original - dsa_original.min()) / (dsa_original.max() - dsa_original.min())
    print(dsa_original.shape, dsa_original.dtype, dsa_original.min(), dsa_original.max())

    dsa_original_uint = np.uint8(dsa_original * 255)
    print(dsa_original_uint.shape, dsa_original_uint.dtype, dsa_original_uint.min(), dsa_original_uint.max())

    #Compute direct warps
    frames = test_data.shape[2]
    direct_matrices = []
    for frameNum in range(0, frames):
        print(f'I am on frame {frameNum}.')
        base_frame = getEightBitImage(test_data,0)
        aligned_frame = getEightBitImage(test_data, frameNum)
        #aligned_frame = np.float32(warp_to_index(test_data, warp_matrices, frameNum, 0))
        direct_matrix = computeWarpMatrixWithFeatures(base_frame, aligned_frame)
        print(direct_matrix)
        direct_matrices.append(direct_matrix)

    # Now let's try using the sequential warping to improve the dsa
    direct_moco_angio = np.zeros_like(test_data)
    for f in range(0,test_data.shape[2]):
        print(f'I am working on frame {f}')
        transformedFrame = apply_warp(test_data, direct_matrices, f)
        direct_moco_angio[:,:,f] = transformedFrame

    print(direct_moco_angio.shape, direct_moco_angio.dtype, direct_moco_angio.min(), direct_moco_angio.max())

    dsa_affine = performDSA(direct_moco_angio)
    dsa_affine = dsa_affine.transpose(2,0,1)
    dsa_affine = dsa_affine.astype('float32')
    dsa_affine = (dsa_affine - dsa_affine.min()) / (dsa_affine.max() - dsa_affine.min())
    print(dsa_affine.shape, dsa_affine.dtype, dsa_affine.min(), dsa_affine.max())

    dsa_affine_uint = np.uint8(dsa_affine * 255)
    print(dsa_affine_uint.shape, dsa_affine_uint.dtype, dsa_affine_uint.min(), dsa_affine_uint.max())

    Ndisp = 7
    indices = np.linspace(0, dsa_original_uint.shape[0]-1, Ndisp).astype(np.int64)
    fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*3,4), constrained_layout=True)
    for i in range(Ndisp):
        axes[i].imshow(dsa_original_uint[indices[i]], cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')
        axes[i].set_title('frm: {}'.format(indices[i]))
    plt.show()

    Ndisp = 7
    indices = np.linspace(0, dsa_affine_uint.shape[0]-1, Ndisp).astype(np.int64)
    fig, axes = plt.subplots(1, Ndisp, figsize=(Ndisp*3,4), constrained_layout=True)
    for i in range(Ndisp):
        axes[i].imshow(dsa_affine_uint[indices[i]], cmap='gray', vmin=0, vmax=255)
        axes[i].axis('off')
        axes[i].set_title('frm: {}'.format(indices[i]))
    plt.show()