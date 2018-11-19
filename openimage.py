# -*- coding: utf-8 -*-
# File: ilsvrc.py

import cv2
import os
import tarfile
import numpy as np
import tqdm
import tensorpack

from tensorpack.utils import logger
# from tensorpack.utils.loadcaffe import get_caffe_pb
from tensorpack.utils.fs import mkdir_p #, download, get_dataset_path
from tensorpack.utils.timer import timed_operation
from tensorpack.dataflow.base import DataFlow, RNGDataFlow

import numpy as np

__all__ = ['OpenImageMeta', 'OpenImage', 'OpenImageFiles']

# CAFFE_OpenImage12_URL = ("http://dl.caffe.berkeleyvisin.org/caffe_ilsvrc12.tar.gz", 17858008)


class OpenImageMeta(object):
    """
    Provide methods to access metadata for OpenImage dataset.
    """

    def __init__(self, path_db):
        #
        self.dir = os.path.expanduser(path_db)
        #
        fname = os.path.join(self.dir, 'annotations', 'labels_used.txt')
        assert os.path.isfile(fname)
        lines = [x.strip().split(',') for x in open(fname).readlines()]

        synset_list = [l[0].strip() for l in lines]
        self.synset_dict = { ii: l for ii, l in enumerate(synset_list) }
        # class-wise weights, considering their frequencies
        cls_weights = np.array([int(l[3].strip()) for l in lines])
        min_w = np.min(cls_weights)
        cls_weights = np.sqrt(min_w / cls_weights)

        self.cls_weights = cls_weights
        # mkdir_p(self.dir)
        # f = os.path.join(self.dir, 'synsets.txt')
        # if not os.path.isfile(f):
        #     self._download_caffe_meta()
        # self.caffepb = None

    # def get_synset_words_1000(self):
    #     """
    #     Returns:
    #         dict: {cls_number: cls_name}
    #     """
    #     fname = os.path.join(self.dir, 'annotations', 'imagelabel_idx2cid.txt')
    #     assert os.path.isfile(fname)
    #     lines = [x.strip().split(',') for x in open(fname).readlines()]
    #     return { int(l[0].strip()): l[1].strip() for l in lines }

    def get_synset_1000(self):
        """
        synset is not 1000 classes any more, but just keeping the name.

        Returns:
            dict: {cls_number: synset_id}
        """
        return self.synset_dict
        # fname = os.path.join(self.dir, 'annotations', 'labels_used.txt')
        # assert os.path.isfile(fname)
        # lines = [x.strip().split(',')[0] for x in open(fname).readlines()]
        # return { ii: l for ii, l in enumerate(lines) }

    @property
    def num_classes(self):
        return len(self.get_synset_1000())

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`OpenImage12.__init__()`.
        Returns:
            list: list of (image filename, positive label, negative label)
        """
        assert name in ['train', 'validation', 'test']

        fn_list = os.path.join(self.dir, 'annotations', '{}_imagelist.txt'.format(name))
        with open(fn_list, 'r') as fh:
            img_names = [f.strip() for f in fh.readlines()]

        res = { n: [[], []] for n in img_names }

        fn_pos = os.path.join(self.dir, 'annotations', '{}_positive_imagelabel.txt'.format(name))
        fn_neg = os.path.join(self.dir, 'annotations', '{}_negative_imagelabel.txt'.format(name))
        assert os.path.isfile(fn_pos), fn_pos
        assert os.path.isfile(fn_neg), fn_neg

        class_dict = self.get_synset_1000()
        inv_dict = {v: k for k, v in class_dict.items()}

        with open(fn_pos) as fh:
            lines = fh.readlines()

        for line in lines:
            datum = line.strip().split(',')
            fname = datum[0].strip()
            if fname not in res:
                continue
            labels = [inv_dict[d.strip()] for d in datum[1:]]
            res[fname][0] += labels

        with open(fn_neg) as fh:
            lines = fh.readlines()

        for line in lines:
            datum = line.strip().split(',')
            fname = datum[0].strip()
            if fname not in res:
                continue
            labels = [inv_dict[d.strip()] for d in datum[1:]]
            res[fname][1] += labels

        ret = [(k, v[0], v[1]) for k, v in res.items()]
        return ret

    # @staticmethod
    # def guess_dir_structure(dir):
    #     """
    #     Return the directory structure of "dir".
    #
    #     Args:
    #         dir(str): something like '/path/to/imagenet/val'
    #
    #     Returns:
    #         either 'train' or 'original'
    #     """
    #     subdir = os.listdir(dir)[0]
    #     # find a subdir starting with 'n'
    #     if subdir.startswith('n') and \
    #             os.path.isdir(os.path.join(dir, subdir)):
    #         dir_structure = 'train'
    #     else:
    #         dir_structure = 'original'
    #     logger.info(
    #         "[OpenImage12] Assuming directory {} has '{}' structure.".format(
    #             dir, dir_structure))
    #     return dir_structure


class OpenImageFiles(RNGDataFlow):
    """
    Same as :class:`OpenImage`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, path_db, name, shuffle=None):
        """
        Same as in :class:`OpenImage`.
        """
        assert name in ['train', 'test', 'validation'], name
        path_db = os.path.expanduser(path_db)
        assert os.path.isdir(path_db), path_db
        self.img_dir = os.path.join(path_db, 'images', name)
        # self.full_dir = os.path.join(path_db, name)
        self.name = name
        assert os.path.isdir(self.img_dir), self.img_dir
        # assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = (name == 'train')
        self.shuffle = shuffle

        # if name == 'train':
        #     dir_structure = 'train'
        # if dir_structure is None:
        #     dir_structure = OpenImageMeta.guess_dir_structure(self.full_dir)

        meta = OpenImageMeta(path_db)
        self.num_classes = meta.num_classes
        self.imglist = meta.get_image_list(name)
        assert len(self.imglist) > 0

        self.cls_weights = meta.cls_weights
        # for fname, _, _ in self.imglist[:10]:
        #     fname = os.path.join(self.img_dir, fname + '.jpg')
        #     if not os.path.isfile(fname):
        #         print(fname)

    def __len__(self):
        return len(self.imglist)

    def __iter__(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, pos_labels, neg_labels = self.imglist[k]
            fname = os.path.join(self.img_dir, fname + '.jpg')
            labels = np.zeros((self.num_classes), dtype=np.float32)
            weights = \
                    np.zeros((self.num_classes), dtype=np.float32) \
                    if neg_labels else \
                    np.zeros((self.num_classes), dtype=np.float32)
            labels[pos_labels] = 1
            weights[pos_labels + neg_labels] = 1
            # class-wise weighting, for handling class imbalance
            weights *= self.cls_weights
            yield fname, labels, weights

    # def get_data(self):
    #     return self.__iter__()


class OpenImage(OpenImageFiles):
    """
    Produces uint8 OpenImage12 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, path_db, name, shuffle=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'validation' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.

        Directory structure:
            path_db/
                images/
                    train/
                    validation/
                annotations/

        """
        super(OpenImage, self).__init__(path_db, name, shuffle)

    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    def __iter__(self):
        for fname, labels, weights in super(OpenImage, self).__iter__():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            yield [im, labels, weights]

    # @staticmethod
    # def get_training_bbox(bbox_dir, imglist):
    #     import xml.etree.ElementTree as ET
    #     ret = []
    #
    #     def parse_bbox(fname):
    #         root = ET.parse(fname).getroot()
    #         size = root.find('size').getchildren()
    #         size = map(int, [size[0].text, size[1].text])
    #
    #         box = root.find('object').find('bndbox').getchildren()
    #         box = map(lambda x: float(x.text), box)
    #         return np.asarray(box, dtype='float32')
    #
    #     with timed_operation('Loading Bounding Boxes ...'):
    #         cnt = 0
    #         for k in tqdm.trange(len(imglist)):
    #             fname = imglist[k][0]
    #             fname = fname[:-4] + 'xml'
    #             fname = os.path.join(bbox_dir, fname)
    #             try:
    #                 ret.append(parse_bbox(fname))
    #                 cnt += 1
    #             except Exception:
    #                 ret.append(None)
    #         logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
    #     return ret



if __name__ == '__main__':
    import cv2
    # meta = OpenImageMeta('./data')
    # print(meta.get_synset_words_1000())

    ds = OpenImageFiles('~/dataset/openimage', 'validation', shuffle=True)
    ds.reset_state()

    for ii, k in enumerate(ds):
        print(k)
        if ii > 20:
            break


    # ds = OpenImage12('/home/wyx/data/fake_ilsvrc/', 'train', shuffle=False)
    # ds.reset_state()
    #
    # for k in ds:
    #     from IPython import embed
    #     embed()
    #     break
