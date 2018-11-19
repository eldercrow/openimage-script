from tensorpack.dataflow import *
import numpy as np
import cv2


class BinaryILSVRC12(dataset.ILSVRC12Files):
    def __iter__(self):
        for fname, label in super(BinaryILSVRC12, self).__iter__():
            try:
                jpeg = cv2.imread(fname)
                h, w = jpeg.shape[:2]
                sf = min(1.0, 448.0 / min(h, w))
                if sf < 1.0:
                    jpeg = cv2.resize(jpeg, (0, 0), fx=sf, fy=sf)
            except e:
                print('Could not process {}'.format(fname))
                continue
            assert np.min(jpeg.shape[:2]) <= 448

            jpeg = cv2.imencode('.jpg', jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tostring()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]


ds0 = BinaryILSVRC12('~/dataset/imagenet', 'val')
ds1 = PrefetchDataZMQ(ds0, nr_proc=8)
LMDBSerializer.save(ds0, './imagenet-val.lmdb')
