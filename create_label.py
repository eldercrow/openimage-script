
# coding: utf-8

# In[1]:


import numpy as np
import csv
import json
import os
from collections import defaultdict


db_type = 'train'


fn_idx2name = 'data/annotation/metadata/imagelabel_idx2cname.txt'
with open(fn_idx2name, 'r') as fh:
    idx2name = [l.strip().split(',') for l in fh.readlines()]
idx2name = { int(l[0].strip()): l[1].strip() for l in idx2name }
name2idx = { v: k for k, v in idx2name.items() }

fn_idx2cid = 'data/annotation/metadata/imagelabel_idx2cid.txt'
with open(fn_idx2cid, 'r') as fh:
    idx2cid = [l.strip().split(',') for l in fh.readlines()]
idx2cid = { int(l[0].strip()): l[1].strip() for l in idx2cid }
cid2idx = { v: k for k, v in idx2cid.items() }

cls_desc = { idx2cid[i]: idx2name[i] for i in idx2cid }

fn_tree = 'data/annotation/metadata/class_tree.json'
with open(fn_tree, 'r') as fh:
    class_tree = json.load(fh)

# In[17]:


fn_label = './data/annotation/raw/{}-annotations-human-imagelabels-boxable.csv'.format(db_type)
pos_labels = defaultdict(list)
neg_labels = defaultdict(list)

with open(fn_label, 'r', newline='') as fh:
    reader = csv.reader(fh)
    for ii, row in enumerate(reader):
        if ii == 0:
            continue
        iid, _, cid, is_pos = row
        cname = cls_desc[cid]
        if int(is_pos) == 1:
            pos_labels[iid].append(cname)
        else:
            neg_labels[iid].append(cname)

pos_labels = dict(pos_labels)
neg_labels = dict(neg_labels)
iid_all = list(set(list(pos_labels.keys()) + list(neg_labels.keys())))


# Remove images not in the image directory.

# In[18]:


path_img = './data/image/{}'.format(db_type)
iid_all = [iid for iid in iid_all if os.path.exists(os.path.join(path_img, iid+'.jpg'))]


# Update positive and negative labels w.r.t the class tree.
# Rules are:
# 1. For a positive label, add its parents as positive.
# 2. For a negative label, add its children as negative.

# In[21]:


def update_labels(labels, class_tree, is_pos):
    res = [l for l in labels]
    for l in labels:
        res += class_tree[l]['parent'] if is_pos else class_tree[l]['children']
    return list(set(res))


# In[22]:


new_pos_labels = {iid: update_labels(labels, class_tree, True)
                  for iid, labels in pos_labels.items()}


# In[23]:


new_neg_labels = {iid: update_labels(labels, class_tree, False)
                  for iid, labels in neg_labels.items()}


# Save labels into the two files, one for positive and one for negative

# In[24]:


fn_imagelist = 'data/annotation/{}_imagelist.txt'.format(db_type)
with open(fn_imagelist, 'w') as fh:
    for iid in iid_all:
        fh.write(iid + '\n')


# In[27]:


fn_positive_labels = 'data/annotation/{}_positive_imagelabel.txt'.format(db_type)
with open(fn_positive_labels, 'w') as fh:
    for k, v in new_pos_labels.items():
        # -1 to name2idx to remove 'entity'
        one_line = '{}, '.format(k) + ', '.join([str(name2idx[vi] - 1) for vi in v])
        fh.write(one_line + '\n')


# In[28]:


fn_negative_labels = 'data/annotation/{}_negative_imagelabel.txt'.format(db_type)
with open(fn_negative_labels, 'w') as fh:
    for k, v in new_neg_labels.items():
        # -1 to name2idx to remove 'entity'
        one_line = '{}, '.format(k) + ', '.join([str(name2idx[vi] - 1) for vi in v])
        fh.write(one_line + '\n')

