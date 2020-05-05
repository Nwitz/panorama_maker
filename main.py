import cv2 as cv
import numpy as np

from Image_Classes import *
from stitching import *

image_counter = 1

# image1 = cv.imread("image_sets/graf/img1.ppm")
# image1 = cv.imread("image_sets/graf/img2.ppm")

# image1 = cv.imread("project_images/Rainier1.png")
# image2 = cv.imread("project_images/Rainier2.png")

# image2 = cv.imread("image_sets/graf/img4.ppm")

#
# image1 = cv.imread("image_sets/yosemite/Yosemite1.jpg")
# image2 = cv.imread("image_sets/yosemite/Yosemite2.jpg")


# Convert image to grey scale
# Blur image to remove noise/false positives
# Return blurred grey scale image
def preprocess(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return grey


def getGradients(image):
    # xGradient = yGradient = xyGradient = np.zeros(image.shape)
    xGradient = cv.Scharr(image, cv.CV_32F, 1, 0)
    yGradient = cv.Scharr(image, cv.CV_32F, 0, 1)
    xyGradient = np.multiply(xGradient, yGradient)
    return xGradient, yGradient, xyGradient


def computeCornerStrengthMatrix(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_x_square = cv.GaussianBlur(cv.multiply(grad_x, grad_x), (3, 3), 1)

    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    grad_y_square = cv.GaussianBlur(cv.multiply(grad_y, grad_y), (3, 3), 1)

    grad_xy = cv.GaussianBlur(cv.multiply(grad_x, grad_y), (3, 3), 1)

    det = cv.subtract(cv.multiply(grad_x_square, grad_y_square), grad_xy ** 2)
    trace = cv.add(grad_x_square, grad_y_square)

    response_raw = det - 0.04 * (trace ** 2)

    ret, response_t = cv.threshold(response_raw, 0, float("inf"), cv.THRESH_TOZERO)
    cv.normalize(response_t, response_t, 0, 255, cv.NORM_MINMAX)

    response_8bit = response_t.astype(np.uint8)
    adaptive = cv.adaptiveThreshold(response_8bit, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, -1)
    return adaptive


def get2DGaussian(side):
    gaussian = cv.getGaussianKernel(side, -1)
    return np.dot(gaussian, np.transpose(gaussian))


def displayCornersOnImage(img):
    thresh = 0
    image = img.image
    corners = img.corner_matrix
    new = image.copy()
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if corners[i][j] > thresh:
                new = cv.circle(new, (j, i), 1, (0, 255, 0), cv.FILLED, 8, 0)
    image_title = "corners on image %s" % image_counter
    print(image_title)
    cv.imshow(image_title, new)
    return new


def harris(image, grey):
    dst = cv.cornerHarris(grey, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow('dst', image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def displayNormalized(title, img):
    dst = img.copy()
    cv.imshow(title, cv.normalize(img, dst, 0, 255, cv.NORM_MINMAX).astype(np.uint8))


# receive 16 by 16 patches
def sift(patch):
    # print("sift")
    # each cell in patch
    descriptor = np.zeros(128)
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    patch_d = 1
    patch_d = cv.normalize(patch, patch_d, 0, 360, cv.NORM_MINMAX)
    desired = 90
    average = np.average(patch_d)
    r_correct = patch_d - (average - desired)
    r_correct = r_correct % 360

    for offset_i in range(4):
        for offset_j in range(4):
            left = offset_i * 4
            right = left + 4
            top = offset_j * 4
            bottom = top + 4
            cell = r_correct[left:right, top:bottom]

            hist = np.histogram(cell, bins)
            starting = (offset_i * 4 + offset_j) * 8
            # print(hist)
            # print(starting)
            descriptor[starting:starting + 8] = hist[0]
            # print(descriptor)
    return descriptor


def extractPatches(imgObj):
    threshold = 100
    features = []
    corner_matrix = imgObj.corner_matrix
    orientation = imgObj.orientation_matrix
    f_count = 0
    keypoints = []
    descriptors = []
    for i in range(8, corner_matrix.shape[0] - 8):
        for j in range(8, corner_matrix.shape[1] - 8):
            if corner_matrix[i][j] > threshold:
                window = orientation[i - 8:i + 8, j - 8:j + 8]
                if f_count % 20 == 0:
                    print("feature:", f_count)
                f_count += 1
                feature = Feature(window, (i, j))
                feature.descriptor = sift(feature.window)
                descriptors.append(feature.descriptor)
                features.append(feature)
                keypoints.append(cv.KeyPoint(feature.coordinate[1], feature.coordinate[0], 16))
    imgObj.__set_features__(features)
    imgObj.__set_keypoints__(keypoints)
    imgObj.__set_descriptors__(np.asarray(descriptors).astype(np.float32))
    return imgObj


# Basic Fast Library for Approximate Nearest Neighbors
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
def matchImagesFlann(image1, image2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(image1.descriptors, image2.descriptors, k=2)
    matches_tresholded = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.80 * n.distance:
            matches_tresholded.append(matches[i][0])

    internal_matches = create_match_objects(image1.keypoints, image2.keypoints, matches_tresholded)

    draw_params = dict(matchColor=None,
                       singlePointColor=(255, 0, 0),
                       matchesMask=None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_image = cv.drawMatches(image1.image, image1.keypoints, image2.image, image2.keypoints, matches_tresholded, None,
                             **draw_params)
    cv.imshow("FLANN match", match_image)


    # internal_matches = create_match_objects(image1.keypoints, image2.keypoints, matches)
    return internal_matches, match_image


def prepare_for_matches(image):
    blurred = preprocess(image)
    Ix, Iy, Ixy = getGradients(blurred)
    imgObj = Image(image, Ix, Iy, Ixy)
    imgObj.__set_corner_matrix__(computeCornerStrengthMatrix(blurred))
    imgObj.__set_orientation_matrix__(np.arctan2(Ix, Iy))
    # displayCornersOnImage(imgObj)
    return extractPatches(imgObj)

def create_match_objects(src_points, dst_points, matches):
    internal_matches = []
    for match in matches:
        queryIdx = match.queryIdx
        trainIdx = match.trainIdx

        src_point = src_points[match.queryIdx].pt
        dst_point = dst_points[match.trainIdx].pt

        internal_match = Match(src_point, dst_point)
        internal_matches.append(internal_match)
    return internal_matches

def process_image(image1, image2):
    iObj1 = prepare_for_matches(image1)
    iObj2 = prepare_for_matches(image2)
    internal_matches, match_image = matchImagesFlann(iObj1, iObj2)
    H = invH = 0
    H, invH, inlier_matches = RANSAC(internal_matches, 0, 150, 50, H, invH, iObj1, iObj2)
    stitch_image = stitch(iObj1, iObj2, H, invH)
    return stitch_image


# boxes, harris save as 1a.png
boxes = cv.imread("project_images/Boxes.png")
blurred = preprocess(boxes)
corners = computeCornerStrengthMatrix(blurred)

image1 = cv.imread("project_images/Rainier1.png")
image2 = cv.imread("project_images/Rainier2.png")
imgObj1 = prepare_for_matches(image1)
image_counter = image_counter + 1
imgObj2 = prepare_for_matches(image2)
r1 = displayCornersOnImage(imgObj1)
r2 = displayCornersOnImage(imgObj2)

cv.imwrite("project_output/1a.png", corners)
cv.imwrite("project_output/1b.png", r1)
cv.imwrite("project_output/1c.png", r2)



# Rainier1.png, Rainier2.png, find matches sift, save as 2.png drawMatches()

internal_matches, match_image = matchImagesFlann(imgObj1, imgObj2)
cv.imwrite("project_output/2.png", match_image)



# Rainier1.png, Rainier2.png, find inlier matches, save as 3.png drawMatches()
H = invH = 0
H, invH, inlier_matches =  RANSAC(internal_matches, 0, 150, 50, H, invH, imgObj1, imgObj2)
cv.imwrite("project_output/3.png", inlier_matches)

# Rainier1.png, Rainier2.png, stitch images, save as 4.png drawMatches()
stitch_image = stitch(imgObj1, imgObj2, H, invH)
cv.imwrite("project_output/4.png", stitch_image)

# loop through last step to find complete panorama
rainier_3 = cv.imread("project_images/Rainier3.png")
rainier_4 = cv.imread("project_images/Rainier4.png")
rainier_5 = cv.imread("project_images/Rainier5.png")
rainier_6 = cv.imread("project_images/Rainier6.png")

stitch_image = process_image(stitch_image, rainier_3)
cv.imwrite("project_output/allStitched_3.png", stitch_image)

stitch_image = process_image(stitch_image, rainier_4)
cv.imwrite("project_output/allStitched_4.png", stitch_image)

stitch_image = process_image(stitch_image, rainier_5)
cv.imwrite("project_output/allStitched_5.png", stitch_image)

stitch_image = process_image(stitch_image, rainier_6)
cv.imwrite("project_output/allStitched_6.png", stitch_image)




# imgObj1 = prepare_for_matches(image1)
# image_counter = image_counter + 1
# imgObj2 = prepare_for_matches(image2)
# internal_matches = matchImagesFlann(imgObj1, imgObj2)
# H = invH = 0
# H, invH = RANSAC(internal_matches, 0, 150, 50, H, invH, imgObj1, imgObj2)
# stitch(imgObj1, imgObj2, H, invH)



cv.waitKey(0)
