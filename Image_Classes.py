class Feature:
    def __init__(self, window, coordinate):
        self.window = window
        self.coordinate = coordinate


class Match:
    def __init__(self, feature_1, feature_2, ssd):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.ssd = ssd

    def compare(self, ssd):
        if (self.ssd >= ssd):
            return False
        else:
            return True

    def set_new_match(self, feature, ssd):
        self.feature_2 = feature
        self.ssd = ssd


class Image:
    def __init__(self, image, Ix, Iy, Ixy):
        self.image = image
        self.Ix = Ix
        self.Iy = Iy
        self.Ixy = Ixy
    def __set_features__(self, features):
        self.features = features  # features are every feature with coordinate and descriptor

    def __set_keypoints__(self, keypoints):
        self.keypoints = keypoints  # keypoints are only features that had a matching keypoint below the threshold

    def __set_descriptors__(self, descriptors):
        self.descriptors = descriptors

    def __set_corner_matrix__(self, corner_matrix):
        self.corner_matrix = corner_matrix

    def __set_orientation_matrix__(self, orientation_matrix):
        self.orientation_matrix = orientation_matrix

class Match:
    def __init__(self, src_point, dst_point):
        self.src_point = src_point
        self.dst_point = dst_point
