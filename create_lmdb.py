import numpy as np
import lmdb, os, sys
import cv2
from multiprocessing import Process, cpu_count, Lock

caffe_root = '/home/hyunjoon/faster-rcnn/caffe-fast-rcnn'

import caffe
from caffe.proto import caffe_pb2


def create_lmdb(fn_lst, root_img, path_lmdb, comp_mean=False):
    '''
    '''
    sz_patch = (320, 320)

    # first convert .lst file to lmdb compatible one.
    fn_lmdb = _convert_lst(fn_lst)

    # if not os.path.exists(path_lmdb):
    #     os.makedirs(path_lmdb)
    # assert os.path.isdir(path_lmdb)

    # then create lmdb
    command_str = caffe_root + '/build/tools/convert_imageset '
    command_str += '--encode_type=jpg '
    command_str += '--encoded=true '
    if sz_patch is not None:
        command_str += '--resize_height=%d ' % sz_patch[0]
        command_str += '--resize_width=%d ' % sz_patch[1]
    command_str += root_img + ' '
    command_str += fn_lmdb + ' '
    command_str += path_lmdb

    os.system(command_str)

    if not comp_mean:
        return

    mean_data_name = os.path.join(path_lmdb, 'mean_img.txt')
    mean_proto_name = os.path.join(path_lmdb, 'mean_img.binaryproto')

    command_str = caffe_root + '/build/tools/compute_image_mean '
    command_str += path_lmdb + ' '
    command_str += mean_proto_name

    os.system(command_str)

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean_proto_name , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    mean_val = np.mean(np.mean(arr, axis=3), axis=2).ravel()

    import ipdb
    ipdb.set_trace()

    print 'Mean image values: ' + str(mean_val)


def _convert_lst(fn_lst):
    #
    path_lst, fn = os.path.split(fn_lst)

    fname, _ = os.path.splitext(fn)
    fn_lmdb = os.path.join(path_lst, fname + '_lmdb.txt')

    with open(fn_lst, 'r') as fh:
        lst_data = fh.read().splitlines()

    with open(fn_lmdb, 'w') as fh:
        for l in lst_data:
            linfo = l.split('\t')
            cid = linfo[1]
            fn = linfo[2]

            lstr = '{} {}\n'.format(fn, cid)
            fh.write(lstr)

    return fn_lmdb


def resample_img_set(fn_lst, root_img, root_res, n_thread=4):
    '''
    Resample original images (crop and resize) within <fn_lst>.
    Result images will be used to create lmdb.
    '''
    with open(fn_lst, 'r') as fh:
        lst_data = fh.read().splitlines()

    lst_data_folds = [lst_data[i::n_thread] for i in range(n_thread)]

    # import ipdb
    # ipdb.set_trace()

    lock = Lock()
    workers = []
    for i in range(n_thread):
        p = Process(target=_process_fold, args=(lst_data_folds[i], root_img, root_res, lock, n_thread, i))
        p.start()
        workers.append(p)

    for i in range(n_thread):
        workers[i].join()


def _process_fold(lst_data, root_img, root_res, lock, n_thread, thread_id):
    #
    for i, lst_row in enumerate(lst_data):
        linfo = lst_row.split('\t')
        fn_img = linfo[2]
        cls_id = linfo[1]

        img = cv2.imread(os.path.join(root_img, fn_img))

        # resize
        sf = 320.0 / min(img.shape[0], img.shape[1])
        ww = int(np.round(img.shape[1] * sf))
        hh = int(np.round(img.shape[0] * sf))
        img = cv2.resize(img, (ww, hh))

        # centre crop
        cx = img.shape[1] / 2
        cy = img.shape[0] / 2
        ll = cx - 160
        rr = cx + 160
        uu = cy - 160
        bb = cy + 160
        assert ll >= 0 and rr <= img.shape[1] and uu >= 0 and bb <= img.shape[0]
        img = img[uu:bb, ll:rr, :]

        res_name = os.path.join(root_res, fn_img)
        res_path, _ = os.path.split(res_name)
        if not os.path.isdir(res_path):
            lock.acquire()
            try:
                if not os.path.exists(res_path):
                    os.makedirs(res_path)
            finally:
                lock.release()
        cv2.imwrite(res_name, img)

        if i % 1000 == 0:
            print 'Fold {}: ({} / {}) images processed.'.format(thread_id, i, len(lst_data))


if __name__ == '__main__':
    #
    fn_lst = '/home/hyunjoon/dataset/openimage/rec_classification/train_openimage_sample.lst'
    root_img = '/home/hyunjoon/dataset/openimage/images/train'
    root_res = '/home/hyunjoon/dataset/openimage/img_resampled/train/'
    path_lmdb = '/home/hyunjoon/dataset/openimage/lmdb/train'

    resample_img_set(fn_lst, root_img, root_res)
    create_lmdb(fn_lst, root_res, path_lmdb, comp_mean=True)
