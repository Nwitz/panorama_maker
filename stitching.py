import numpy as np
import cv2 as cv
import random


def project(x1, y1, H):
    src_point = np.array([x1, y1, 1])

    dst_point = np.matmul(H, np.transpose(src_point))
    dst_point = np.transpose(dst_point)
    dst_point = dst_point / dst_point[2]
    x2 = dst_point[0]
    y2 = dst_point[1]
    return np.array([x2, y2])


def computeInlierCount(H, matches, numMatches, inlierThreshold):
    numMatches = 0
    inlier_matches = []
    dMatches = []
    for i, match in enumerate(matches):
        projection = project(match.src_point[0], match.src_point[1], H)
        dist = np.linalg.norm(match.dst_point - projection)
        if dist < inlierThreshold:
            numMatches += 1
            inlier_matches.append(match)
            dMatches.append(cv.DMatch(i, i, dist))

    return numMatches, inlier_matches, dMatches


# matches -> contains source_pts and dst_points
def RANSAC(matches, numMatches, numIterations, inlierThreshold, hom, homInv, image1Display, image2Display):
    # Randomly select 4 matches
    final_inlier_matches = []
    for i in range(numIterations):
        selected = random.sample(matches, 4)
        src_points, train_points = get_points(selected)
        H, status = cv.findHomography(src_points, train_points, 0)
        current_matches, inlier_matches, d_matches = computeInlierCount(H, matches, 0, inlierThreshold)
        # if new highest matches, save number, and matches for next homography.
        if current_matches > numMatches:
            numMatches = current_matches
            final_inlier_matches = inlier_matches
            hom = H

    # Use the most matches outcome to computer new homography
    src_points, dst_points = get_points(final_inlier_matches)
    hom, _ = cv.findHomography(src_points, dst_points)
    homInv, _ = cv.findHomography(dst_points, src_points)
    current_matches, inlier_matches, d_matches = computeInlierCount(hom, final_inlier_matches, 0, inlierThreshold)

    src_kpts = convert_to_keypoints(src_points)
    dst_kpts = convert_to_keypoints(dst_points)


    draw_params = dict(matchColor=None,
                       singlePointColor=(255, 0, 0),
                       matchesMask=None,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    inlier_matches = cv.drawMatches(image1Display.image, src_kpts, image2Display.image, dst_kpts, d_matches, None,
                             **draw_params)
    #draw matches here. must convert points to keypoints
    cv.imshow("RANSAC", inlier_matches)
    return hom, homInv, inlier_matches

def stitch(image1, image2, hom, homInv):
    print("stitch")
    image1_offset, stitch_image = allocate_stitch_image(image1, image2, homInv)
    add_image1_to_stitched(image1, stitch_image, image1_offset)
    cv.imwrite("saved_images/stitch_im1.jpg", stitch_image)
    add_image2_to_stitched(image2, stitch_image, hom, image1_offset)
    cv.imshow("reference stitch", stitch_image)
    cv.imwrite("saved_images/stitch_full.jpg", stitch_image)
    return stitch_image



def add_image1_to_stitched(image, stitch_image, offset):
    tl = offset
    shape = image.image.shape
    br = np.zeros(2, int)
    br[0] = (shape[0] + offset[0]).astype(int)
    br[1] = (shape[1] + offset[1]).astype(int)

    stitch_image[tl[0]:br[0], tl[1]:br[1], :] = image.image
    cv.imshow("stitch", stitch_image)

def add_image2_to_stitched(image, stitch_image, hom, offset):
    stitch_shape = stitch_image.shape
    image_shape = image.image.shape
    for i in range(stitch_shape[0]):
        if i % 50 == 0:
            print("add image 2 - row:", i)
        for j in range(stitch_shape[1]):
            point = project(j - offset[1], i - offset[0], hom)
            contribution = get_contribution(image.image, (point[0], point[1]))
            if in_range(image_shape, point):
                if np.sum(stitch_image[i, j, :]) == 0:
                    stitch_image[i, j, :] = contribution
                else:
                    contribution = np.add(np.int32(stitch_image[i, j, :]), np.int32(contribution)) / 2.0
                    stitch_image[i, j, :] = np.uint8(contribution)

def in_range(image_shape, point):
    return point[1] > 0 and point[0] > 0 and point[1] < image_shape[0] and point[0] < image_shape[1]

def get_contribution(image, point):
    patch = cv.getRectSubPix(image, (1, 1), point)
    return patch



def allocate_stitch_image(image1, image2, homInv):
    top_left, bottom_right = get_extrema(image1, image2, homInv)
    #top left will be an offset
    x = np.ceil(bottom_right[0] - top_left[0]).astype(int)
    y = np.ceil(bottom_right[1] - top_left[1]).astype(int)
    stitch_image = np.zeros((y, x, 3), np.uint8)
    # return offset to know correct positioning for image1
    temp = top_left[1]
    top_left[1] = top_left[0]
    top_left[0] = temp
    offset = np.multiply(-1, top_left).astype(int)
    return offset, stitch_image


# extract the extrema in both images, this will be the new image
def get_extrema(image1, image2, homInv):
    print("get_extrema")
    corners = []
    shape1 = image1.image.shape
    shape2 = image2.image.shape
    corners.append(project(       0,          0, homInv))
    corners.append(project(       0,  shape2[0], homInv))
    corners.append(project(shape2[1], shape2[0], homInv))
    corners.append(project(shape2[1],         0, homInv))
    topLeft = [0, 0]
    bottomRight = [shape1[1], shape1[0]]
    for corner in corners:
        x = corner[0]
        y = corner[1]
        if (x < topLeft[0]):
            topLeft[0] = x
        if (x > bottomRight[0]):
            bottomRight[0] = x
        if (y < topLeft[1]):
            topLeft[1] = y
        if (y > bottomRight[1]):
            bottomRight[1] = y
    return topLeft, bottomRight



def convert_to_keypoints(points):
    keypoints = []
    for point in points:
        keypoints.append(cv.KeyPoint(point[0], point[1], 0))
    return keypoints

def get_points(matches):
    src_points = []
    dst_points = []

    for match in matches:
        src_points.append(match.src_point)
        dst_points.append(match.dst_point)
    return np.float32(src_points), np.float32(dst_points)