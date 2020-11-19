import glob
import sys,os
#import argparse

import numpy as np

import cv2
'''
def parse_args():
        parser = argparse.ArgumentParser(description="cse 473/573 project2")
        parser.add_argument(
            type=str, default="../extra2/",
            help="path to the images")
        args = parser.parse_args()
        return args
'''
def get_homography(point_group_1, point_group_2):
    P = []
    for i in range(0, len(point_group_1)):
        point_1_x, point_1_y = point_group_1[i][0], point_group_1[i][1]
        point_2_x, point_2_y = point_group_2[i][0], point_group_2[i][1]
        P.append([point_1_x, point_1_y, 1, 0, 0, 0, -point_2_x*point_1_x, -point_2_x*point_1_y, -point_2_x])
        P.append([0, 0, 0, point_1_x, point_1_y, 1, -point_2_y*point_1_x, -point_2_y*point_1_y, -point_2_y])
    P = np.asarray(P)
    U, S, Vh = np.linalg.svd(P)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H
    
def ransac(matches, key_point1, key_point2):
    """
    Input - Matches from one_NN_matcher(..)
    Returns - Inlier homography matrix
    """    
    threshold=5.0
    #4 random matches considered
    # Store best homography 
    best_H_matrix = None
    best_matches = []
    inliers = -1
        
    for curr_iter in range(10): # 10 iterations
        # Randomly permute matches list
        np.random.shuffle(matches)
        for i in range(0, len(matches), 4): # consider 4 matches per iteration
            if i + 4 >= len(matches):#termination condition
                break;
            point_group_1, point_group_2 = [], []
            for match in matches[i : i + 4]:
                point_group_1.append(key_point1[match[0]].pt)
                point_group_2.append(key_point2[match[1]].pt)
            # Compute Homography
            H = get_homography(point_group_1, point_group_2)

            # Initialize inlier Count
            count = 0
            # Get matches of current iteration
            iter_matches = []
            # Add inliers; remove outliers
            for match in matches:
                # Get point on left image
                point_1 = [key_point1[match[0]].pt[0], key_point1[match[0]].pt[1], 1]
                # Get actual point on right image
                point_2_actual = [key_point2[match[1]].pt[0], key_point2[match[1]].pt[1]]
                # Predict point
                point_2_predicted = np.dot(H, point_1)
                point_2_predicted = (point_2_predicted / point_2_predicted[-1])
                point_2_predicted = point_2_predicted[:-1]
                # Calculate distances between predicted & actual point
                diff = np.linalg.norm(np.subtract(point_2_actual, point_2_predicted))
                # inlier_count
                if diff < threshold:
                    count += 1
                    iter_matches.append(match)
            # Calculate best model
            if count > inliers:
                inliers = count
                best_H_matrix = H
                best_matches = iter_matches
                # Terminate early based on count
                if inliers > int(0.3 * len(key_point1)):
                    return best_H_matrix, best_matches
    return best_H_matrix, best_matches
    
def get_distances(key_point1, key_point2, desc1, desc2, width):
        """
        Utility function to get left & right distances
        """
        matches = matcher(desc1, desc2)
        _, matches = ransac(matches, key_point1, key_point2)

        matches = sorted(matches, key=lambda x: x[2])
        distances = []
        
        for match in matches:
            point1, point2 = key_point1[match[0]].pt, key_point2[match[1]].pt
            point1, point2 = np.array(point1), np.array(point2)

            point2[0] += width

            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)
        return distances

def get_direction(key_point1, key_point2, desc1, desc2, img1, img2):
    """
    input keypoints and descriptors of two images along with images
    Returns - Order
    """
    
    height, width, y = img2.shape
        
    left_distances = get_distances(key_point1, key_point2, desc1, desc2, width)
    right_distances = get_distances(key_point2, key_point1, desc2, desc1, width)
    
    if np.mean(left_distances) <= np.mean(right_distances):
        return -1
    return 1

def matcher(des1, des2, k=2):
    """
    Input - 
            des1 & des2 - Descriptor matrices for 2 images
            k - Number of nearest neighbors to consider
    Returns - A vector of nearest neighbors of des1 & their indices for keypoints
    
    Mnemonic - des1 is like xtest, des2 is like xtrain
    """

    # Compute the L2 equations
    distances = np.sum(des1 ** 2, axis=1, keepdims=True) + np.sum(des2 ** 2, axis=1) - 2 * des1.dot(des2.T)
    distances = np.sqrt(distances)
    #print(distances)
    
    
    # Get smallest indices 
    min_indices = np.argsort(distances, axis=1)
    
    # Init ndarray 
    nearest_neighbors = []
    
    # Iter for nearest neighbors
    for i in range(min_indices.shape[0]):
        neighbors = min_indices[i][:k]
        #print(neighbors)
        curr_matches = []
        for j in range(len(neighbors)):
            match=[]
            match.append(i)
            match.append(neighbors[j])
            match.append(distances[i][neighbors[j]] * 1.)
            #match = cv2.DMatch(i, neighbors[j], 0, distances[i][neighbors[j]] * 1.)
            curr_matches.append(match)
        nearest_neighbors.append(curr_matches)
    good_matches = []
    # Iter through matches
    for m, n in nearest_neighbors:
        # Perform ratio threshold
        if m[2] < 0.75 * n[2]:
            good_matches.append(m)
    return good_matches


class Image:
    
    #Each Image object consists of the image, keypoints & descriptors.
    def __init__(self, img, kp=None, des=None):
        self.img = img
        self.shape = img.shape
        sift = cv2.xfeatures2d.SIFT_create()
        self.key_point, self.descriptor = sift.detectAndCompute(self.img, None)


def stitch(folder):
    #Reads the images from the given path, aligns them and writes result into panorama.jpg
    # Get image paths in given directory
    image_paths = glob.glob(os.path.join(folder,"*.jpg"))

    # Read all these images
    images = [cv2.imread(image_path) for image_path in image_paths if "panorama" not in image_path]

    # Convert each of these images to Image objects
    for i, image in enumerate(images):
        images[i] = Image(image)
        
    images_list = [images[0]] 
    
    #sorting the images according to distances
    for k in range(1, len(images)):
        y = images[k]
        count = 0
        value = False
        x=len(images_list)
        for j in range(0,x):
            x = images[j]
            order = get_direction(x.key_point, y.key_point,x.descriptor, y.descriptor,x.img, y.img)
            if(order == 1):
                images_list.insert(count, y)
                value = True
                break;
            count+=1
        if(value == False):
            images_list.append(y)
    
    def warp_two_images(image1, image2, H_matrix):
        
        #warp img2 to img1 with homograph H
        height_image_1, weight_image_1 = image1.shape[:2]
        height_image_2, weight_image_2 = image2.shape[:2]
        points_image_1 = np.float32([[0, 0], [0, height_image_1], [weight_image_1, height_image_1], [weight_image_1, 0]]).reshape(-1, 1, 2)
        points_image_2 = np.float32([[0, 0], [0, height_image_2], [weight_image_2, height_image_2], [weight_image_2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(points_image_2, H_matrix)
        points_joint = np.concatenate((points_image_1, pts2_), axis=0)
        xmin, ymin = np.int32(points_joint.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(points_joint.max(axis=0).ravel() + 0.5)
        least = [-xmin, -ymin]
        Ht = np.array(
            [
                [1, 0, least[0]], 
                [0, 1, least[1]], 
                [0, 0, 1]
            ]
        )
        #translate image2 w.r.t image1
        result = cv2.warpPerspective(image2.img, Ht.dot(H_matrix),(xmax - xmin, ymax - ymin))
        result[least[1]: height_image_1 + least[1], least[0]: weight_image_1 + least[0]] = image1.img
        return result

    def stitch_2_images(image1, image2):
        # Convert images to Image objects
        if not isinstance(image1, Image):
            image1 = Image(image1)
            
        if not isinstance(image2, Image):
            image2 = Image(image2)
        # Get best matches from matcher function
        matches = matcher(image1.descriptor, image2.descriptor)
        
        #Checking the no. of matches
        if(len(matches) / len(image1.key_point) > 0.1):

        # Best Homography matrix
            H_matrix, r = ransac(matches, image1.key_point, image2.key_point)

        # Create panorama
            panorama = warp_two_images(image1, image2, np.linalg.inv(H_matrix))
        else:
            panorama = image2.img
        return panorama

    # If 2 images, construct panorama
    print("No.of images",len(images_list))
    if len(images_list) == 2:
        panorama = stitch_2_images(images_list[1], images_list[0])
    
    # If 3 images, construct 2 panoramas &
    # return final one
    else:
        panorama = stitch_2_images(images_list[1], images_list[0])
        for k in range(2,len(images_list)):
            panorama = stitch_2_images(images_list[k],panorama)
    return panorama

def main():
    cv2.imwrite(os.path.join(sys.argv[1],"panorama.jpg"), stitch(sys.argv[1]))

if __name__ == "__main__":
    main()
    