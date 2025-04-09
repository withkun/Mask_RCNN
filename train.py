# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import yaml
import imgaug

from PIL import Image
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = osp.join(ROOT_DIR, 'logs')

# Local path to trained weights file
COCO_MODEL_PATH = osp.join(ROOT_DIR, 'mask_rcnn_coco.h5')
# Download COCO trained weights from Releases if needed
if not osp.exists(COCO_MODEL_PATH):
     utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'shapes'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    TRAIN_ROIS_PER_IMAGE = 500
    MAX_GT_INSTANCES = 500
    DETECTION_MAX_INSTANCES = 500

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 30

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    LEARNING_RATE = 0.0001


class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme_export_json生成的label_names.txt文件, 从而得到mask每一层对应的实例标签
    def from_text_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            labels = f.readlines()
            labels = [line.strip() for line in labels]
            del labels[0]   # remove _background_
        return labels

    # 解析labelme_export_json生成的info.yaml文件, 从而得到mask每一层对应的实例标签(Deprecated)
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        if info['yaml_path'].endswith('.txt'):
            return self.from_text_get_class(image_id)

        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]   # remove _background_
        return labels

    # labelme_export_json生成的label.png图像是单通道的索引图.
    def draw_mask(self, num_obj, mask, image, image_id):
        # print('draw_mask-->', image_id)
        # print('self.image_info', self.image_info)
        info = self.image_info[image_id]
        # print('info-->', info)
        # print('info[width]----->', info['width'], '-info[height]--->', info['height'])
        for x in range(info['width']):
            for y in range(info['height']):
                # print('image_id-->', image_id, '-x--->', x, '-y--->', y)
                # print('info[width]----->', info['width'], '-info[height]--->', info['height'])
                at_pixel = image.getpixel((x, y))
                if at_pixel == 0:   # 背景
                    continue
                # HWC格式图像, 将对应层的对应位置赋值1
                mask[y, x, at_pixel - 1] = 1

        mask_path = osp.splitext(info['mask_path'])[0] + '.npz'
        np.savez_compressed(mask_path, mask)
        info['mask_path'] = mask_path
        return mask

    # 重新写load_shapes, 里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path、yaml_path
    # dataset_root_path = 'dateset'
    # img_folder  = osp.join(dataset_root_path, 'img')
    # yaml_folder = osp.join(dataset_root_path, 'info')
    # mask_folder = osp.join(dataset_root_path, 'label')
    def load_shapes(self, img_list, idx_list, img_folder, mask_folder, yaml_folder):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class('shapes', 1, 'stone')

        for i, idx in enumerate(idx_list):
            filename, extension = osp.splitext(img_list[idx])
            # print('id-->', idx, ' img_list[', idx, ']-->', img_list[idx], 'filename-->', filename)
            if osp.exists(osp.join(mask_folder, filename + '.npz')):
                mask_path = osp.join(mask_folder, filename + '.npz')
            else:
                mask_path = osp.join(mask_folder, filename + '.png')
            yaml_path = osp.join(yaml_folder, filename + '.yaml')
            cv_img = cv2.imread(osp.join(img_folder, img_list[idx]))
            self.add_image('shapes', image_id=i, path=osp.join(img_folder, img_list[idx]),
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # print('in load mask image_id', image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        if not info['mask_path'].endswith('.npz'):
            img = Image.open(info['mask_path'])
            num_obj = self.get_obj_index(img)
            mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
            mask = self.draw_mask(num_obj, mask, img, image_id)     # 相当耗性能
        else:
            mask = np.load(info['mask_path'])['arr_0']

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find('stone') != -1:
                labels_form.append('stone')
            # elif labels[i].find('leg') != -1:
            #     # print 'leg'
            #     labels_form.append('leg')
            # elif labels[i].find('well') != -1:
            #     # print 'well'
            #     labels_form.append('well')
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# All train data set
dataset_root_path = r'/data/Mask_RCNN/datasets/coco_data'

img_folder  = osp.join(dataset_root_path, 'img')
mask_folder = osp.join(dataset_root_path, 'label')
yaml_folder = osp.join(dataset_root_path, 'info')

img_list = os.listdir(img_folder)
img_count = len(img_list)
img_id_list = np.arange(img_count)
random.shuffle(img_id_list)

# train与valid数据集拆分
train_num = int(img_count * 0.75)
valid_num = img_count - train_num
train_idx = img_id_list[:train_num]
valid_idx = img_id_list[train_num:train_num+valid_num]

# Train data
dataset_train = DrugDataset()
dataset_train.load_shapes(img_list, train_idx, img_folder, mask_folder, yaml_folder)
dataset_train.prepare()
print('dataset_train-->', train_idx)

# Valid data
dataset_valid = DrugDataset()
dataset_valid.load_shapes(img_list, valid_idx, img_folder, mask_folder, yaml_folder)
dataset_valid.prepare()
print('dataset_valid-->', valid_idx)

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

config = ShapesConfig()
config.display()

# Create model in training mode
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = 'coco'  # imagenet, coco, or last

if init_with == 'imagenet':
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == 'coco':
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # print(COCO_MODEL_PATH)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
elif init_with == 'last':
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers='heads' freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

aug = imgaug.augmenters.Sequential([              # 建立一个名为Seq的实例, 定义增强方法, 用于增强
    imgaug.augmenters.Fliplr(0.5),                # 对%50的图像进行做左右翻转
    imgaug.augmenters.Flipud(0.5),                # 对%50的图像进行做上下翻转
    imgaug.augmenters.Multiply((0.8, 1.25)),      # 亮度变化, 乘以(0.8, 1.25)随机采样一个值
])

# 在预训练模型的基础上训练分为两步:
print('Start heads training...')
model.train(dataset_train, dataset_valid,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads', augmentation=aug)

# Fine tune all layers
# Passing layers='all' trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
print('Start fine tune training...')
model.train(dataset_train, dataset_valid,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=50,
            layers='all', augmentation=aug)

