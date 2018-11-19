import os
import cv2
import numpy as np
from multiprocessing import Pool
from functools import partial

def resize_and_write(datum, path_src, path_dst):
    index, prefix_img = datum
    fn_img = os.path.join(path_src, prefix_img + '.jpg')
    assert os.path.isfile(fn_img), fn_img

    fn_res = os.path.join(path_dst, prefix_img + '.jpg')

    img = cv2.imread(fn_img)
    h, w = img.shape[:2]

    sf = 448.0 / min(h, w)
    if sf < 1.0:
        img = cv2.resize(img, (0, 0), fx=sf, fy=sf)
    assert min(img.shape[0], img.shape[1]) <= 448

    if index % 1000 == 0:
        print('Processed {}\'th image'.format(index))
    # print(fn_res)
    cv2.imwrite(fn_res, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

if __name__ == '__main__':
    #
    db_type = 'train'

    path_src = '/local/openimage/images/{}'.format(db_type)
    path_dst = '/local/hyunjoon/dataset/openimage/resized/{}'.format(db_type)
    if not os.path.isdir(path_dst):
        os.makedirs(path_dst)

    f = partial(resize_and_write, path_src=path_src, path_dst=path_dst)

    fn_list = '/local/hyunjoon/dataset/openimage/annotations/{}_imagelist.txt'.format(db_type)
    list_img = [(ii, l.strip()) for ii, l in enumerate(open(fn_list, 'r').read().splitlines())]

    print('Total number of images to process = {}'.format(len(list_img)))
    p = Pool(6)
    p.map(f, list_img)
