import sys, os
import csv
from collections import defaultdict

import ipdb


def subsample_class(root_db, fn_classes, fn_annot, fn_result, ext='.jpg'):
    '''
    fn_classes
        Google openimage dataset v3 provides a list of trainable classes.

    fn_annot
        The human verified annotation file is a .csv file, containing
            (image name, verification type, class id, pos(1)/neg(0).
            e.g. 000595fe6fee6369,verification,/m/0j7ty,0

    fn_result
        I will reorder them to create .lst file as follows
            ID  class_id (0 to 4999)    image_name
    '''
    with open(fn_classes, 'r') as fh:
        classes = fh.read().splitlines()

    img_positive_dict = defaultdict(list)
    img_negative_dict = defaultdict(list)

    # class name to ID
    clsname2id = {n: i for i, n in enumerate(classes)}

    with open(fn_annot, 'r') as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            try:
                img_name, _, class_id, class_prob = row
            except:
                print 'Could not parse row {}: {}'.format(i, str(row))

            if class_id not in clsname2id:
                continue

            if int(class_prob) == 1:
                img_positive_dict[img_name].append(clsname2id[class_id])
            else:
                img_negative_dict[img_name].append(clsname2id[class_id])

            if i % 100000 == 0:
                print 'Processing {} rows.'.format(i)

    # maximum number of classes assigned to each image
    max_cls = 0
    for v in img_positive_dict.values():
        max_cls = max(max_cls, len(v))

    with open(fn_result, 'w') as fh:
        for ii, (img_name, cls_ids) in enumerate(img_positive_dict.items()):
            datum = '{}\t'.format(ii)
            datum += ''.join('{}\t'.format(v) for v in cls_ids)
            datum += ''.join('-1\t' for _ in range(max_cls-len(cls_ids)))
            datum += img_name + ext + '\n'

            fh.write(datum)


if __name__ == '__main__':
    #
    root_db = './data'
    fn_classes = './data/annotation/classes-trainable.txt'
    fn_annot = './data/annotation/train/annotations-human.csv'
    fn_result = './data/annotation/train/train.lst'

    subsample_class(root_db, fn_classes, fn_annot, fn_result)
