"""
field_parser.py
parser for field data

"""
from __future__ import print_function

import os
import struct
import numpy as np


def parse_field_all(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of internal field and boundary
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_internal_field_content(content), parse_boundary_content(content)


def parse_internal_field(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of internal field
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_internal_field_content(content)


def parse_internal_field_content(content):
    """
    parse internal field from content
    :param content: contents of lines
    :return: numpy array of internal field
    """
    is_binary = is_binary_format(content)
    for ln, lc in enumerate(content):
        if lc.startswith(b'internalField'):
            if b'nonuniform' in lc:
                return parse_data_nonuniform(content, ln, len(content), is_binary)
            elif b'uniform' in lc:
                return parse_data_uniform(content[ln])
            break
    return None


def parse_boundary_field(fn):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :return: numpy array of boundary field
    """
    if not os.path.exists(fn):
        print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        return parse_boundary_content(content)


def parse_boundary_content(content):
    """
    parse each boundary from boundaryField
    :param content:
    :return:
    """
    data = {}
    is_binary = is_binary_format(content)
    bd = split_boundary_content(content)
    for boundary, (n1, n2) in bd.items():
        pd = {}
        n = n1
        while True:
            lc = content[n]
            if b'nonuniform' in lc:
                v = parse_data_nonuniform(content, n, n2, is_binary)
                pd[lc.split()[0]] = v
                if not is_binary:
                    n += len(v) + 4
                else:
                    n += 3
                continue
            elif b'uniform' in lc:
                pd[lc.split()[0]] = parse_data_uniform(content[n])
            n += 1
            if n > n2:
                break
        data[boundary] = pd
    return data


def parse_data_uniform(line):
    """
    parse uniform data from a line
    :param line: a line include uniform data, eg. "value           uniform (0 0 0);"
    :return: data
    """
    if b'(' in line:
        return np.array([float(x) for x in line.split(b'(')[1].split(b')')[0].split()])
    return float(line.split(b'uniform')[1].split(b';')[0])


def parse_data_nonuniform(content, n, n2, is_binary):
    """
    parse nonuniform data from lines
    :param content: data content
    :param n: line number
    :param n2: last line number
    :param is_binary: binary format or not
    :return: data
    """
    num = int(content[n + 1])
    if not is_binary:
        if b'scalar' in content[n]:
            data = np.array([float(x) for x in content[n + 3:n + 3 + num]])
        else:
            data = np.array([ln[1:-2].split() for ln in content[n + 3:n + 3 + num]], dtype=float)
    else:
        nn = 1
        if b'vector' in content[n]:
            nn = 3
        elif b'symmtensor' in content[n]:
            nn = 6
        elif b'tensor' in content[n]:
            nn = 9
        buf = b''.join(content[n+2:n2+1])
        vv = np.array(struct.unpack('{}d'.format(num*nn),
                                    buf[struct.calcsize('c'):num*nn*struct.calcsize('d')+struct.calcsize('c')]))
        if nn > 1:
            data = vv.reshape((num, nn))
        else:
            data = vv
    return data


def split_boundary_content(content):
    """
    split each boundary from boundaryField
    :param content:
    :return: boundary and its content range
    """
    bd = {}
    n = 0
    in_boundary_field = False
    in_patch_field = False
    current_path = ''
    while True:
        lc = content[n]
        if lc.startswith(b'boundaryField'):
            in_boundary_field = True
            if content[n+1].startswith(b'{'):
                n += 2
                continue
            elif content[n+1].strip() == b'' and content[n+2].startswith(b'{'):
                n += 3
                continue
            else:
                print('no { after boundaryField')
                break
        if in_boundary_field:
            if lc.rstrip() == b'}':
                break
            if in_patch_field:
                if lc.strip() == b'}':
                    bd[current_path][1] = n-1
                    in_patch_field = False
                    current_path = ''
                n += 1
                continue
            if lc.strip() == b'':
                n += 1
                continue
            current_path = lc.strip()
            if content[n+1].strip() == b'{':
                n += 2
            elif content[n+1].strip() == b'' and content[n+2].strip() == b'{':
                n += 3
            else:
                print('no { after boundary patch')
                break
            in_patch_field = True
            bd[current_path] = [n,n]
            continue
        n += 1
        if n > len(content):
            if in_boundary_field:
                print('error, boundaryField not end with }')
            break

    return bd


def is_binary_format(content, maxline=20):
    """
    parse file header to judge the format is binary or not
    :param content: file content in line list
    :param maxline: maximum lines to parse
    :return: binary format or not
    """
    for lc in content[:maxline]:
        if b'format' in lc:
            if b'binary' in lc:
                return True
            return False
    return False
