import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import glob
from tensorflow.keras.utils import load_img, img_to_array

global rootdir, image_paths, target_size
rootdir = '.' 
images_directory = os.path.join(rootdir,'dataset/celeba') 
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):

    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def run_dlib_shape(image):
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    for (i, rect) in enumerate(rects):

        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image
############################################################################################


def extract_features_labels(image_directory, label_path, Objective):
    with open(os.path.join(label_path, labels_filename), 'r') as labels_file:
        lines = labels_file.readlines()
    image_filenames = glob.glob(os.path.join(image_directory, '*'))
    
    if Objective == "gender":
        shape_labels = {line.split(',')[0]: int(line.split(',')[1]) for line in lines[1:]}
    else:
        shape_labels = {line.split(',')[0]: int(line.split(',')[2]) for line in lines[1:]}
    
    all_features, all_labels, file_no = [], [], []
    
    counter = 0
    while counter < len(image_filenames):
        img_path = image_filenames[counter]
        file_name = os.path.split(img_path)[-1]
        img = img_to_array(load_img(img_path, target_size=None, interpolation='bicubic'))
        features, _ = run_dlib_shape(img)
        if features is not None:
            all_features.append(features)
            all_labels.append(shape_labels[file_name])
        else:
            file_no.append(file_name)
            print(f"{file_name} not included")
        counter += 1
    print(file_no)
    
    landmark_features = np.array(all_features)
    shape_labels = np.array(all_labels)
    
    return landmark_features, shape_labels