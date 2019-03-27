import numpy as np
from six.moves import cPickle
import cv2
import uuid
import os


def extractImagesAndLabels(path, file):
    with open(path + file, 'rb') as f:
        data_dict = cPickle.load(f, encoding='latin1')
        images = data_dict['data']
        labels = data_dict['labels']

        imagearray = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        labelarray = np.array(labels)
        print(imagearray.shape)
        print(len(imagearray))
        return imagearray, labelarray


def extractCategories(path, file):
    with open(path + file, 'rb') as f:
        dict = cPickle.load(f)
    return dict['label_names']


def saveCifarImage(array, path, file_name):
    # array is RGB. cv2 needs BGR
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    file_path = os.path.join(path, file_name)
    # print(file_path)
    cv2.imwrite(file_path, array)
    return

# imgarray, lblarray = extractImagesAndLabels("/home/lst/datasets/cifar-10-batches-py/", f"data_batch_{str(1)}")
# print(imgarray.shape)
# print(lblarray.shape)
# for i in range(1, 6):
#     imgarray, lblarray = extractImagesAndLabels("/home/lst/datasets/cifar-10-batches-py/", f"data_batch_{str(i)}")
#     # print(imgarray.shape)
#     # print(lblarray.shape)
#
#     categories = extractCategories("/home/lst/datasets/cifar-10-batches-py/", "batches.meta")
#     # print(categories)
#     cats = []
#     for i in range(len(imgarray)):
#         image_dir = '/home/lst/datasets/cifar-10-images/' + categories[lblarray[i]]
#
#         os.makedirs(image_dir, exist_ok=True)
#         identifier = uuid.uuid4()
#         file_name = f"{identifier}.png"
#         saveCifarImage(imgarray[i], image_dir, file_name)

imgarray, lblarray = extractImagesAndLabels("/home/lst/datasets/cifar-10-batches-py/", "test_batch")
# print(imgarray.shape)
# print(lblarray.shape)

categories = extractCategories("/home/lst/datasets/cifar-10-batches-py/", "batches.meta")
# print(categories)
for i in range(len(imgarray)):
    image_dir = '/home/lst/datasets/cifar-10-images_test/' + categories[lblarray[i]]

    os.makedirs(image_dir, exist_ok=True)
    identifier = uuid.uuid4()
    file_name = f"{identifier}.png"
    saveCifarImage(imgarray[i], image_dir, file_name)
