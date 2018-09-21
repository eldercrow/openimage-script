
# coding: utf-8

# In[1]:


import numpy as np
import csv
import json
import os
from collections import defaultdict


# In[2]:


def load_label_hierarchy(fn_tree):
    '''
    Load and parse label tree.
    '''
    with open(fn_tree, 'r') as fh:
        json_data = json.loads(fh.read())


# Load class hierarchy and name data.

# In[3]:


with open('./data/annotation/raw/bbox_labels_600_hierarchy.json', 'r') as fh:
    json_data = fh.read()
class_hierarchy = json.loads(json_data)

with open('./data/annotation/raw/class-descriptions-boxable.csv', 'r') as fh:
    desc_data = fh.read().splitlines()

desc_data = [c.split(',') for c in desc_data]
cls_desc = {c[0].strip(): c[1].strip() for c in desc_data}
inv_cls_desc = {c[1].strip(): c[0].strip() for c in desc_data}

cls_desc['/m/0bl9f'] = 'Entity'
inv_cls_desc['Entity'] = '/m/0bl9f'


# In[4]:


def convert_entry(entry, cls_desc):
    cls_name = cls_desc[entry['LabelName']]
    res_entry = {}
    for childname in ['Subcategory', 'Part']:
        if childname in entry:
            res_entry[childname] = [convert_entry(e, cls_desc) for e in entry[childname]]
    res_entry['LabelName'] = cls_name
    return res_entry


# In[5]:


converted_hierarchy = convert_entry(class_hierarchy, cls_desc)


# Give a class ID to each class, and build a hierarchy-based positive negative binary label map.

# In[6]:


def list_names(entry, name_list=[]):
    name_list.append(entry['LabelName'])
    for childname in ['Subcategory', 'Part']:
        if childname in entry:
            for e in entry[childname]:
                list_names(e, name_list)


# In[26]:


nlist = []
list_names(converted_hierarchy, nlist)

# keep 'entity' in the index 0
name_list = nlist[0:1] + list(set(nlist[1:]))

name2idx = {k: name_list.index(k) for k in name_list}
idx2name = {name_list.index(k): k for k in name_list}

fn_idx2name = 'data/annotation/metadata/imagelabel_idx2cname.txt'
with open(fn_idx2name, 'w') as fh:
    for k, v in idx2name.items():
        one_line = '{}, {}\n'.format(k, v)
        fh.write(one_line)

fn_idx2cid = 'data/annotation/metadata/imagelabel_idx2cid.txt'
with open(fn_idx2cid, 'w') as fh:
    for k, v in idx2name.items():
        one_line = '{}, {}\n'.format(k, inv_cls_desc[v])
        fh.write(one_line)


# Build the class tree list.
# Each entry in the list contains its parent(s) and children.

# In[8]:


def build_tree(entry, class_tree, inv_cls_desc, pname):
    cname = entry['LabelName']
    parent = pname + [cname] if cname is not 'Entity' else pname
    children = []

    for subname in ['Subcategory', 'Part']:
        if subname in entry:
            for e in entry[subname]:
                children.append(e['LabelName'])
                build_tree(e, class_tree, inv_cls_desc, parent)

    class_tree[cname]['parent'] += pname
    class_tree[cname]['children'] += children
    class_tree[cname]['class_id'].append(inv_cls_desc[cname])


# In[9]:


class_tree = {cname: {'class_id': [], 'parent': [], 'children': []} for cname in name_list}
build_tree(converted_hierarchy, class_tree, inv_cls_desc, [])
# remove duplications
class_tree = {k: {vk: list(set(vv)) for vk, vv in v.items()}
              for k, v in class_tree.items()}


fn_tree = 'data/annotation/metadata/class_tree.json'
with open(fn_tree, 'w') as fh:
    json.dump(class_tree, fh)

