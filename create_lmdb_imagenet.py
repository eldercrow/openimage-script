from tensorpack import *
from tensorpack.dataflow import *
import numpy as np
import cv2
import os

# class BinaryILSVRC12(dataset.ILSVRC12Files):
#     def __iter__(self):
#         for fname, label in super(BinaryILSVRC12, self).__iter__():
#             try:
#                 jpeg = cv2.imread(fname)
#                 h, w = jpeg.shape[:2]
#                 sf = min(1.0, 448.0 / min(h, w))
#                 if sf < 1.0:
#                     jpeg = cv2.resize(jpeg, (0, 0), fx=sf, fy=sf)
#             except e:
#                 print('Could not process {}'.format(fname))
#                 continue
#             assert np.min(jpeg.shape[:2]) <= 448
#
#             jpeg = cv2.imencode('.jpg', jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tostring()
#             jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
#             yield [jpeg, label]


class BinaryDataSet(DataFlow):
    def __init__(self, filename_labels, default_size=224, min_scale=0.4):
        self._data = filename_labels
        self._max_sz = int(np.round(default_size / min_scale))

    def __len__(self):
        return len(self._data)

    def reset_state(self):
        pass

    def __iter__(self):
        for fname, label in self._data:
            try:
                jpeg = cv2.imread(fname)
                h, w = jpeg.shape[:2]
                sf = min(1.0, self._max_sz / np.sqrt(h*w))
                if sf < 1.0:
                    jpeg = cv2.resize(jpeg, (0, 0), fx=sf, fy=sf)
            except e:
                print('Could not process {}'.format(fname))
                continue
            # assert np.min(jpeg.shape[:2]) <= self._max_sz

            jpeg = cv2.imencode('.jpg', jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tostring()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, int(label)]


def create_aspect_data(db_path, db_type):
    #
    ds = dataset.ILSVRC12Files(db_path, db_type, shuffle=False)
    aspect_data = []
    for ii, (fname, label) in enumerate(ds):
        if ii % 10000 == 9999:
            print(ii)
        try:
            jpeg = cv2.imread(fname)
            h, w = jpeg.shape[:2]
            aspect_data.append((fname, int(label), h/w))
        except e:
            print('Could not process {}'.format(fname))
        # if ii > 10:
        #     break

    aspect_data = sorted(aspect_data, key=lambda k: k[2])
    return aspect_data


# def create_lmdb(filenames, fn_lmdb):
#     #


asp_path = './'
db_path = '~/dataset/imagenet'
db_type = 'val'

fn_aspect = os.path.join(asp_path, 'aspect_data_{}.txt'.format(db_type))

try:
    aspect_data = open(fn_aspect, 'r').read().splitlines()
    filename_labels = [l.split(',')[0:2] for l in aspect_data]
except:
    print('Could not read aspect info from {}, creating one...'.format(fn_aspect))
    aspect_data = create_aspect_data(db_path, db_type)
    with open(fn_aspect, 'w') as fh:
        for d in aspect_data:
            fh.write('{},{},{:.4f}\n'.format(d[0], d[1], d[2]))
    filename_labels = [l[0:2] for l in aspect_data]
ds0 = BinaryDataSet(filename_labels)
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
LMDBSerializer.save(ds0, './imagenet-aspect-{}.lmdb'.format(db_type))
