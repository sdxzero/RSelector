import keras
import numpy as np
import tensorflow as tf
import time
import os
import random
import math
from PIL import Image
import cv2
import mrcz
from scipy import ndimage
import threading
import progressbar
import zignor
from keras import metrics
import keras_resnet

import pyximport
pyximport.install()
from compute_overlap import compute_overlap

def read_box(box_file):
    boxes = []
    with open(box_file,'r') as fd:
        for line in fd:
            if line.startswith('#'):
                print(line)
                continue
            split = line.rstrip().split()
            if len(split)!=4:
                continue
            x1,y1,width,height = [int(float(i)) for i in split]
            x2 = x1+width
            y2 = y1+height
            x1 = max(x1,0)
            y1 = max(y1,0)

            boxes.append((x1,y1,x2,y2))
    return boxes

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_dataset_pairs(directory):
    mrc_files = [m for m in os.listdir(directory) if m.endswith('.mrc')]
    box_files = [b for b in os.listdir(directory) if b.endswith('.box')]

    pairs = []
    for m_file in mrc_files:
        m_idx = m_file.rstrip('.mrc')
        b_pair = m_idx+'.box'
        if b_pair in box_files:
            pairs.append((os.path.join(directory,m_file),os.path.join(directory,b_pair)))
    return pairs


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    if path.endswith('.mrc'):
        img,_ = mrcz.readMRC(path,slices=0)
        img = img.astype(np.float32)
        img = (img - np.min(img))/(np.max(img)-np.min(img))*255
        img = img.astype(np.uint8)
        image = cv2.equalizeHist(img)
        # image = image[:,:,np.newaxis]
        image  =np.array(Image.fromarray(img).convert('RGB'))
        # if len(image.shape)==2:
        #     image = image[:,:,np.newaxis]
    else:
        image = np.asarray(Image.open(path).convert('RGB'))
    return image[:,:,::-1]
    # else:
    #     image = np.asarray(Image.open(path))
    #     image = image[:,:,np.newaxis]
    #     assert len(image.shape)==3
    #     return image

def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape

def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes

def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.6,
    positive_overlap=0.8
):
    """ Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # overlaps = compute_overlap(anchors, annotations)
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
    # ignore_indices = (max_overlaps < negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets

def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    negative_overlap=0.6,
    positive_overlap=0.7
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
    # print(negative_overlap)
    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch  = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     = 1

            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1]     = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch

def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def preprocess_image(x, mode='tf'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())


    if mode == 'tf':
        x[..., 0] -= 127.5
        x[..., 1] -= 127.5
        x[..., 2] -= 127.5
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
    elif mode == 'grey':
        img_max =  np.max(x)
        img_min = np.min(x)
        x = (x-img_min)/(img_max-img_min)*255

        flag = 1
        if flag==1:
            x /=127.5
            x -=1
        elif flag==2:
            x /= 255


    return x

def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """

    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    return output

def compute_resize_scale(image_shape, min_side=1024, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    if len(image_shape)==3:
        (rows, cols,_) = image_shape
    elif len(image_shape)==2:
        (rows,cols) = image_shape
    else:
        raise ValueError("image_shape error")

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale

def resize_image(img, min_side=1024, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def translation(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation 2D vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

def change_transform_origin(transform, center):
    """ Create a new transform representing the same transformation,
        only with the origin of the linear part changed.
    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        translate(center) * transform * translate(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])

def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result

def transform_aabb(transform, aabb):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1:        The minimum x value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum x value of the AABB.
        y2:        The maximum y value of the AABB.
    Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]

def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    anchor_params=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5  ]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        # all_anchors = np.vstack((all_anchors,shifted_anchors))
        # all_anchors = tf.concat([all_anchors,shifted_anchors],axis=0)

    return all_anchors

class Augmentation(object):

    def __init__(self, is_grey=False,is_jitter=0):
        self.is_grey = is_grey
        self.is_jitter = is_jitter


    def image_augmentation(self, image):


        # if random.random()<=self.is_jitter:
        #     image = self.jitter(image)
#       augmentations = [self.additive_gauss_noise, self.add, self.contrast_normalization, self.multiply, self.sharpen, self.dropout]
        augmentations = [
            # self.empty,
                         self.additive_gauss_noise,
                         self.add,
                         self.contrast_normalization,
                         self.multiply,
                         self.sharpen,
                        ]
        if np.random.rand()> 0.5:
            augmentations.append(self.gauss_blur)
        else:
            augmentations.append(self.avg_blur)

    #version1
        selected_augs = random.sample(augmentations, np.random.randint(1,len(augmentations)))
        image = image.astype(np.float32, copy=False)
        # if random.random()<=ratio:
        # for sel_aug in selected_augs:
        for sel_aug in augmentations:

            image = sel_aug(image)

    #version2
        # selected_augs = random.sample(augmentations,1)[0]
        # # if random.random()<=ratio:
        # image = selected_augs(image)

         #   print "Mean after", sel_aug, " sum: ", np.mean(image)
        # if self.is_grey:
        #     min_img = np.min(image)
        #     max_img = np.max(image)
        #     image = ((image - min_img)/(max_img-min_img))*255
        # #    image = np.clip(image, 0, 255)
        #     image = image.astype(np.uint8, copy=False)

        return image


    def empty(self,image):
        return image

    def gauss_blur(self, image, sigma_range=(0,3)):
        rand_sigma = sigma_range[0] + np.random.rand() * (sigma_range[1] - sigma_range[0])
        result = ndimage.gaussian_filter(image, sigma=rand_sigma, mode="nearest")

        if not np.issubdtype(image.dtype,np.float32):
            result = result.astype(np.float32, copy=False)
        return result


    def avg_blur(self, image, kernel_size=(2,7)):
        rang_kernel_size = np.random.randint(kernel_size[0], kernel_size[1])
        image = ndimage.uniform_filter(image, size=rang_kernel_size, mode='nearest')
        return image


    def sharpen(self, image, alpha=(0.0, 1.0), lightness=(0.75, 2.0)):

        rand_alpha = alpha[0] + np.random.rand() * (alpha[1] - alpha[0])
        rand_lightness = lightness[0] + np.random.rand() * (lightness[1] - lightness[0])
        no_chang_mat = np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]])

        u = -1.0
        edges_mat = np.array([[u, u, u],
                              [u, 8.0+rand_lightness, u],
                              [u,u, u]]) * 1.0/9.0

        final_mat = (1-rand_alpha)*no_chang_mat + rand_alpha*edges_mat

        # so much faster than the alternatives...
        image = cv2.filter2D(image, -1, final_mat)

        return image

    def additive_gauss_noise(self, image, sigma_range_factor=0.05):

        if len(image.shape)==2:
            image = image[:,:,np.newaxis]
        width = 2*3*np.std(image)
        max_sigma = width*sigma_range_factor
        rand_sigma = np.random.rand()*max_sigma

        noise = zignor.randn(image.shape[0], image.shape[1])
        np.multiply(noise, rand_sigma,out=noise)
        noise = noise[:,:,np.newaxis]
        np.add(image, noise, out=image)

        if not np.issubdtype(image.dtype, np.float32):
            image = image.astype(np.float32, copy=False)

        return image


    def contrast_normalization(self, image, alpha=(0.5,2.0)):

        rand_multiplyer = alpha[0] + np.random.rand() * (alpha[1] - alpha[0])

        middle = np.median(image)
        np.subtract(image, middle, out=image)
        np.multiply(rand_multiplyer, image, out=image)
        np.add(middle, image, out=image)

        return image

    def dropout(self, image, ratio=(0.01,0.1)):
        if isinstance(ratio, float):
            rand_ratio = ratio
        else:
            rand_ratio = ratio[0] + np.random.rand() * (ratio[1] - ratio[0])
        mean_val = np.mean(image)
        drop = np.random.binomial(n=1, p=1-rand_ratio, size=(image.shape[0], image.shape[1]))
        image[drop == 0] = mean_val

        return image

    def add(self, image, scale=0.05):

        width = 2*3*np.std(image)
        width_rand = scale * width
        rand_constant = (np.random.rand()*width_rand)-width_rand/2
        np.add(image, rand_constant, out=image)

        return image

    def multiply(self, image, range=(0.5, 1.5)):

        rand_multiplyer = range[0] + np.random.rand()*(range[1]-range[0])
        np.multiply(image, rand_multiplyer, out=image)
        return image

    def jitter(self,image):

        # scale the image
        w = image.shape[0]
        h = image.shape[1]
        scale = np.random.uniform() / 10. + 1.
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # translate the image
        max_offx = (scale - 1.) * w
        max_offy = (scale - 1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)

        image = image[offy: (offy + h), offx: (offx + w)]

        # flip the image
        flip_selection = np.random.randint(0, 4)
        flip_vertical = flip_selection == 1
        flip_horizontal = flip_selection == 2
        flip_both = flip_selection == 3

        if flip_vertical:
            image = cv2.flip(image, 1)
        if flip_horizontal:
            image = cv2.flip(image, 0)
        if flip_both:
            image = cv2.flip(image, -1)

        return image

class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


class Generator(object):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=1024,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config=None
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config

        self.group_index = 0
        self.lock        = threading.Lock()
        self.aug = Augmentation()
        self.group_images()

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            # print(annotations['bboxes'],image.shape)
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                # warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                #     group[index],
                #     image.shape,
                #     annotations['bboxes'][invalid_indices, :]
                # ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations

        imgaug_ratio = 0.3
        boxaug_ratio = 0
        if random.random()<=imgaug_ratio:
            image = self.aug.image_augmentation(image)

        # elif imgaug_ratio<random.random()<=boxaug_ratio+imgaug_ratio:
        #     box_mask = np.ones_like(image)
        #     for anno in annotations['bboxes']:
        #         xmin,ymin,xmax,ymax = anno
        #         try:
        #             image[ymin:ymax,xmin:xmax,:] = self.aug.image_augmentation(image[ymin:ymax,xmin:xmax,:])
        #         except:
        #             pass
            # image = np.multiply(image,box_mask)



        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):

            image_batch[image_index, :image.shape[0], :image.shape[1],:image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        # if self.config and 'anchor_parameters' in self.config:
        #     anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)
        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes(),
            negative_overlap=0.3,
            positive_overlap=0.6
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        start = time.time()
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)
        end = time.time()
        # print("compute input consumed time:{:<.2f}".format(end-start))
        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)



class MrcGenerator(Generator):

    def __init__(self,pairs,**kwargs):

        self.image_list = []
        self.annos_list = []
        self.classes = {'particle':0}
        self.labels = {0:'particle'}
        for mrc_file,box_file in pairs:
            self.image_list.append(mrc_file)
            self.annos_list.append(box_file)
        super(MrcGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_list)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def has_label(self, label):
        return label in self.labels

    def has_name(self, name):
        return name in self.classes

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self,image_index):
        return self.image_list[image_index]

    def image_aspect_ratio(self, image_index):
        image = read_image_bgr(self.image_path(image_index))
        ratio = float(image.shape[0])/image.shape[1]
        return ratio

    def load_image(self, image_index):
        # print("\nmrc_generator load image :{}".format(image_index))
        img =  read_image_bgr(self.image_path(image_index))

        return img

    def load_annotations(self, image_index):
        anno_file = self.annos_list[image_index]
        boxes = read_box(anno_file)
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        annotations['labels'] = np.zeros((len(boxes),))
        annotations['bboxes'] = np.array(boxes,dtype=np.float)


        return annotations

def vgg_retinanet(num_classes, backbone='vgg16', fpn=True,inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        channel = 3
        inputs = keras.layers.Input(shape=(None, None, channel))

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False,weights=None)
        # vgg = VGG16(input_tensor=inputs,include_top=False,weights=None)
        layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    elif backbone == 'vgg19':
        vgg = keras.applications.VGG19(input_tensor=inputs,include_top=False,weights=None)
        layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_outputs = [vgg.get_layer(name).output for name in layer_names]
    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs,fpn=fpn, **kwargs)

def resnet_retinanet(num_classes, backbone='resnet50', fpn=True,inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    return retinanet(inputs=inputs, num_classes=num_classes, fpn=fpn,backbone_layers=resnet.outputs[1:], **kwargs)


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.image.resize_images(source, (target_shape[2], target_shape[3]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result

class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
#    sizes   = [32, 64, 128, 256, 512],
#    strides = [8, 16, 32, 64, 128],
    sizes   = [32, 64, 176,
               # 256, 512
               ],
    strides = [8, 16, 44,
               # 64, 128
               ],
    ratios  = np.array([ 1], keras.backend.floatx()),
    scales  = np.array([1.05,1.17,1.37,2.1], keras.backend.floatx()),
)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

# def shift(shape, stride, anchors):
#     """ Produce shifted anchors based on shape of the map and stride size.
#
#     Args
#         shape  : Shape to shift the anchors over.
#         stride : Stride to shift the anchors with over the shape.
#         anchors: The anchors to apply at each location.
#     """
#
#     # create a grid starting from half stride from the top left corner
#     try:
#         shift_x = (np.arange(0, shape[1]) + 0.5) * stride
#         shift_y = (np.arange(0, shape[0]) + 0.5) * stride
#
#         shift_x, shift_y = np.meshgrid(shift_x, shift_y)
#
#         shifts = np.vstack((
#             shift_x.ravel(), shift_y.ravel(),
#             shift_x.ravel(), shift_y.ravel()
#         )).transpose()
#     except:
#         print("shift() error:1228")
#         # shift_x = tf.add(tf.range(0,shape[1],1.0),0.5)*stride
#         # shift_y = tf.add(tf.range(0,shape[0],1.0),0.5)*stride
#         #
#         # shift_x,shift_y = tf.meshgrid(shift_x,shift_y)
#         # shift_x = tf.reshape(shift_x,[1,-1])
#         # shift_y = tf.reshape(shift_y,[1,-1])
#         # shifts = tf.stack([
#         #     shift_x, shift_y,
#         #     shift_x, shift_y
#         # ],axis=0)
#         # tf.transpose(shifts)
#
#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     A = anchors.shape[0]
#     K = shifts.shape[0]
#     anchors_rs = tf.reshape(anchors,[1,A,4])
#     shifts_t = tf.transpose(tf.reshape(shifts,[1,K,4]),[1,0,2])
#     # all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
#     all_anchors = anchors_rs+shifts_t
#     # all_anchors = all_anchors.reshape((K * A, 4))
#     all_anchors = tf.reshape(all_anchors,[K*A,4])
#     return all_anchors

def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

def shift_tensor(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes

class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        if keras.backend.image_data_format() == 'channels_first':
            anchors = shift_tensor(features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = shift_tensor(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config

class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config

class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())
        if keras.backend.image_data_format() == 'channels_first':
            height = shape[2]
            width  = shape[3]
        else:
            height = shape[1]
            width  = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=512,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)
    # outputs = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(outputs)
    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    # outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper




    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    return [P3, P4,  P5 ]



def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """

    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])



def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)

def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 400,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = keras.backend.max(classification, axis    = 1)
        labels  = keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)
    other_              = [keras.backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations

def _get_detections(generator, model, score_threshold=0.05, max_detections=400, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # if save_path is not None:
        #     draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)
        #
        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_precision_recall(generator,all_annotations,all_detections,beta=2):
    score = {}
    crange = np.arange(0.1,1,0.01)
    max_f1 = 0
    max_conf = 0
    max_prec = 0
    max_rec = 0
    max_iou = 0

    TP_beta = 0
    anno_beta = 0
    det_beta = 0
    f1_best = 0
    f1_conf_best = 0
    f1_precision_best = 0
    f1_recall_best = 0

    fbeta_best = 0
    fbeta_conf_best = 0
    fbeta_precision_best = 0
    fbeta_recall_best = 0
    for confidence in crange:
        try:
            confidence = round(confidence,2)
            for label in range(generator.num_classes()):
                if not generator.has_label(label):
                    continue

                TP_sum=0
                iou_sum = 0
                total_annos = 0
                total_detects = 0
                precision_sum = 0
                recall_sum = 0
                f1_sum = 0



                for i in range(generator.size()):
                    TP = 0
                    iou = 0
                    detecs = all_detections[i][label]
                    annos = all_annotations[i][label]

                    detecs = detecs[np.where(detecs[:,4]>=confidence)]
                    num_annos = len(annos)
                    num_detects = len(detecs)
                    total_annos += num_annos
                    total_detects += num_detects
                    for ann in annos:
                        overlap  = compute_overlap(np.expand_dims(ann,axis=0),detecs)
                        try:
                            assigned_detection = np.argmax(overlap,axis=1)
                            max_overlap = overlap[0,assigned_detection]
                            if max_overlap>=0.5:
                                TP +=1
                                iou += max_overlap
                        except:
                            pass

                    FP = num_detects - TP
                    FN = num_annos - TP

                    # if num_detects==0:
                    #     precision = 0
                    # else:
                    #     precision = TP/num_detects
                    # recall = TP/num_annos
                    # f1 = 2*precision*recall/(precision+recall+0.00001)
                    if TP==0:
                        iou= 0
                    else:
                        iou = iou/TP

                    TP_sum +=TP
                    # total_annos += num_annos
                    # total_detects += num_detects
                    # precision_sum += precision
                    # recall_sum += recall
                    # f1_sum += f1
                    # iou_sum += iou

                TP_beta = TP_sum
                precision = TP_beta/total_detects
                recall = TP_beta/total_annos
                f1 = 2*precision*recall/(precision+recall+0.00001)
                fbeta = (1+beta**2)*precision*recall/(beta**2*precision+recall+0.00001)
                if f1>f1_best:
                    f1_conf_best = confidence
                    f1_precision_best = precision
                    f1_recall_best = recall
                    f1_best = f1
                if fbeta>fbeta_best:
                    fbeta_conf_best = confidence
                    fbeta_precision_best = precision
                    fbeta_recall_best = recall
                    fbeta_best = fbeta

                precision_mean = precision_sum/generator.size()
                recall_mean = recall_sum/generator.size()
                f1_mean = f1_sum/generator.size()
                iou_mean = iou_sum/generator.size()

                # if f1_mean>max_f1:
                #     max_f1 = f1_mean
                #     max_conf = confidence
                #     max_rec = recall_mean
                #     max_prec = precision_mean
                #     max_iou = iou_mean
        except:
            pass

    print("F1_max: confidence:{}  precision:{:<.3f} recall:{:<.3f} f1:{:<.3f}".format(f1_conf_best, f1_precision_best, f1_recall_best,  f1_best))
    if beta!=1:
        print("F{}_max: confidence:{}  precision:{:<.3f} recall:{:<.3f} f{}:{:<.3f}".format(beta,fbeta_conf_best,fbeta_precision_best,fbeta_recall_best,beta,fbeta_best))
            # print("confidence_threshold:{:<.2f} TP:{}  precision:{:<.4f} recall:{:<.4f} f1:{:<.4f} iou:{:<.3f}".format(confidence,float(TP),float(precision),float(recall),float(f1),float(iou_mean)))
    return fbeta_best

def evaluate(
    generator,
    model,
    beta=2,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=400,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    # average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    f_beta_max = get_precision_recall(generator, all_annotations, all_detections,beta=beta)

    # for label in range(generator.num_classes()):
    #     if not generator.has_label(label):
    #         continue
    #
    #     false_positives = np.zeros((0,))
    #     true_positives  = np.zeros((0,))
    #     scores          = np.zeros((0,))
    #     num_annotations = 0.0
    #
    #     for i in range(generator.size()):
    #         detections           = all_detections[i][label]
    #         annotations          = all_annotations[i][label]
    #         num_annotations     += annotations.shape[0]
    #         detected_annotations = []
    #
    #         for d in detections:
    #             scores = np.append(scores, d[4])
    #
    #             if annotations.shape[0] == 0:
    #                 false_positives = np.append(false_positives, 1)
    #                 true_positives  = np.append(true_positives, 0)
    #                 continue
    #
    #             overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
    #             assigned_annotation = np.argmax(overlaps, axis=1)
    #             max_overlap         = overlaps[0, assigned_annotation]
    #
    #             if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
    #                 false_positives = np.append(false_positives, 0)
    #                 true_positives  = np.append(true_positives, 1)
    #                 detected_annotations.append(assigned_annotation)
    #             else:
    #                 false_positives = np.append(false_positives, 1)
    #                 true_positives  = np.append(true_positives, 0)
    #
    #     if num_annotations == 0:
    #         average_precisions[label] = 0, 0
    #         continue
    #
    #     indices         = np.argsort(-scores)
    #     false_positives = false_positives[indices]
    #     true_positives  = true_positives[indices]
    #
    #     false_positives = np.cumsum(false_positives)
    #     true_positives  = np.cumsum(true_positives)
    #
    #     recall    = true_positives / num_annotations
    #     precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    #
    #     average_precision  = _compute_ap(recall, precision)
    #     average_precisions[label] = average_precision, num_annotations

    return f_beta_max

class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 500,
        parallel_iterations   = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config

class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        beta=2,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=300,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose
        self.beta = beta

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # run evaluation
        f_beta_max = evaluate(
            self.generator,
            self.model,
            beta=self.beta,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # total_instances = []
        # precisions = []
        # for label, (average_precision, num_annotations ) in average_precisions.items():
        #     if self.verbose == 1:
        #         print('{:.0f} instances of class'.format(num_annotations),
        #               self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        #     total_instances.append(num_annotations)
        #     precisions.append(average_precision)
        # if self.weighted_average:
        #     self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        # else:
        #     self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        #
        # if self.tensorboard is not None and self.tensorboard.writer is not None:
        #     import tensorflow as tf
        #     summary = tf.Summary()
        #     summary_value = summary.value.add()
        #     summary_value.simple_value = self.mean_ap
        #     summary_value.tag = "mAP"
        #     self.tensorboard.writer.add_summary(summary, epoch)
        #
        # logs['mAP'] = self.mean_ap
        #
        # if self.verbose == 1:
        #     print('mAP: {:.4f}'.format(self.mean_ap))
        logs['Fscore'] = f_beta_max


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    fpn=True,
    num_anchors             = None,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    if fpn:
        features = create_pyramid_features(C3, C4, C5)
        pyramids = __build_pyramid(submodels, features)
    else:
        f = features[0]
        pyramids = [default_regression_model(4, num_anchors, name='regression')(f),
               default_classification_model(num_classes, num_anchors, name="classification")(f)]

    # for all pyramid levels, run available submodels


    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)

def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    # else:
    #     assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3',
                                                              'P4',
                                                              'P5',
                                                              # 'P6', 'P7'
                                                              ]]
    anchors  = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]


#version 1
    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)




    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)

def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels         = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


class RModel:
    def __init__(self):



        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


        self.model = None
        self.training_model = None
        self.predict_model = None

        self.train_generator = None
        self.valid_generator = None

        self.callbacks = None
        self.anchorparm =  AnchorParameters(
#    sizes   = [32, 64, 128, 256, 512],
#    strides = [8, 16, 32, 64, 128],
                sizes   = [32, 64, 176,
               # 256, 512
               ],
                strides = [8, 16, 44,
               # 64, 128
               ],
                ratios  = np.array([1], keras.backend.floatx()),
                scales  = np.array([1,1.5,2], keras.backend.floatx()),
                )


    def set_session(self,gpu):
        if gpu==-1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        keras.backend.tensorflow_backend.set_session(get_session())

    def set_generator(self,train_directory,valid_directory=None,ratio=0.25):
        if valid_directory:
            train_pairs = create_dataset_pairs(train_directory)
            valid_pairs = create_dataset_pairs(valid_directory)
        else:
            all_pairs = create_dataset_pairs(train_directory)
            train_size = int(len(all_pairs)*(1-ratio))
            train_pairs = all_pairs[:train_size]
            valid_pairs = all_pairs[train_size:]

        train_generator = MrcGenerator(train_pairs)
        valid_generator = MrcGenerator(valid_pairs)

        return train_generator,valid_generator

    def set_callback(self,model,predict_model,valid_generator,model_save_name='default'):
        callbacks = []
        evaluation = Evaluate(valid_generator)
        evaluation = RedirectModel(evaluation, predict_model)
        callbacks.append(evaluation)

        model_name = '{}.h5'.format(model_save_name)
        checkpoint = keras.callbacks.ModelCheckpoint(
            model_name,
            verbose=1,
            save_best_only=True,
            monitor="Fscore",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=2,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        ))
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='Fscore',
            patience=2,
            verbose=0,
            mode='max'
        ))

        return callbacks


    def set_model(self,num_classes,backbone='vgg19',fpn=True,weights=None,alpha=1):
        if backbone=='vgg16':
            model = vgg_retinanet(num_classes,backbone='vgg16',fpn=fpn)
        elif backbone=='vgg19':
            model = vgg_retinanet(num_classes,backbone='vgg19',fpn=fpn)
        # elif backbone=='resnet50':
        #     model = resnet_retinanet(num_classes,backbone='resnet50')

        if weights:
            model.load_weights(weights,by_name=True,skip_mismatch=True)
        training_model = model
        predict_model = retinanet_bbox(model=model, anchor_params=None)
        training_model.compile(
            loss={
                'regression': smooth_l1(),
                'classification': focal()
            },
            optimizer=keras.optimizers.adam(lr=0.0001, epsilon=1e-8, clipnorm=0.001),
        )

        return model, training_model, predict_model

    def load(self,weights):
        if self.model == None:
            self.model, self.training_model, self.predict_model = self.set_model(num_classes=1, weights=weights)
        else:
            self.model.load_weights(weights,by_name=True,skip_mismatch=True)
        print("Weights loaded ...")

    def save(self,save_name):
        self.model.save(save_name)
        print("Model save done:{}".format(save_name))


    def train(self,train_directory,valid_directory=None,valid_ratio=0.25,gpu=0,weights=None,save_path="model.h5",epochs=7,steps=2000,param=None):

        # self.train_generator,self.valid_generator = self.set_generator(train_directory,valid_directory)
        # self.train_generator.compute_shapes = make_shapes_callback(self.model)
        # self.valid_generator.compute_shapes = self.train_generator.compute_shapes

        if not param:
            alpha = param['alpha']
            backbone = param['backbone']
            max_det = param['max_det']
        else:
            alpha = 1
            backbone = 'vgg19'
            max_det = 400
        self.model, self.training_model, self.predict_model = self.set_model(num_classes=1,backbone=backbone,weights=weights,alpha=alpha)

        self.train_generator, self.valid_generator = self.set_generator(train_directory, valid_directory,valid_ratio)
        print("#"*10+'\n'+"Images using to train network:{}\nImages using to validate model:{}\n".format(self.train_generator.size(),self.valid_generator.size())+'#'*10)
        self.train_generator.compute_shapes = make_shapes_callback(self.model)
        self.valid_generator.compute_shapes = self.train_generator.compute_shapes


        self.callbacks = self.set_callback(self.model,self.predict_model,self.valid_generator,save_path)

        history = self.training_model.fit_generator(generator=self.train_generator,steps_per_epoch=steps,epochs=epochs,callbacks=self.callbacks)

    def predict(self,micrograph_directory,threshold,nms=True,output='Boxes'):

        self.predict_model = retinanet_bbox(self.model,nms=nms,class_specific_filter=True)
        total_time = 0
        predict_mrcs = [os.path.join(micrograph_directory,m) for m in os.listdir(micrograph_directory) if m.endswith(('.mrc','.jpg'))]
        for idx,mrc in enumerate(predict_mrcs):
            start = time.time()
            mrc_base_index = os.path.basename(mrc)[:-4]
            image = read_image_bgr(mrc)
            image = image.copy()
            image,scale = resize_image(image)
            # image = np.expand_dims(image,axis=2)
            boxes,scores,labels = self.predict_model.predict_on_batch(np.expand_dims(image,axis=0))
            boxes = boxes/scale
            # print(boxes.shape,scores.shape)
            if not os.path.exists(output):
                os.mkdir(output)
            fd_box = open(os.path.join(output,mrc_base_index+'.box'),'w')
            for b,s in zip(boxes[0],scores[0]):
                if s>float(threshold):
                    xmin,ymin,xmax,ymax = b
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    w = xmax - xmin
                    h = ymax - ymin
                    fd_box.write('{}\t{}\t{}\t{}\n'.format(xmin,ymin,w,h))
            fd_box.close()
            finish = time.time()
            time_used = finish - start
            total_time +=time_used
            print("Index:{} Micrograph:{} coordinates write done,time used {:<.2f} seconds".format(idx,mrc,time_used))
        print("#"*10+'\n'+'Prediction done.Mean time used:{:<.2f} seconds'.format(total_time/len(predict_mrcs)))


