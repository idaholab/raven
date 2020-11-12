"""
mesh_parser.py
parse mesh data from constant/polymesh
"""
from __future__ import print_function

import numpy as np
import os
import struct
from collections import namedtuple
from field_parser import parse_internal_field, is_binary_format

Boundary = namedtuple('Boundary', 'type, num, start, id')


def is_integer(s):
    try:
        x = int(s)
        return True
    except ValueError:
        return False


class FoamMesh(object):
    """ FoamMesh class """
    def __init__(self, path):
        self.path = os.path.join(path, "constant/polyMesh/")
        self._parse_mesh_data(self.path)
        self.num_point = len(self.points)
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(self.owner)
        self._set_boundary_faces()
        self._construct_cells()
        self.cell_centres = None
        self.cell_volumes = None
        self.face_areas = None

    def read_cell_centres(self, fn):
        """
        read cell centres coordinates from data file,
        the file can be got by `postProcess -func 'writeCellCentres' -time 0'
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.cell_centres = parse_internal_field(fn)

    def read_cell_volumes(self, fn):
        """
        read cell volumes from data file,
        the file can be got by `postProcess -func 'writeCellVolumes' -time 0'
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.cell_volumes = parse_internal_field(fn)

    def read_face_areas(self, fn):
        """
        read face areas from data file,
        :param fn: cell centres file name, eg. '0/C'
        :return: None
        """
        self.face_areas = parse_internal_field(fn)

    def cell_neighbour_cells(self, i):
        """
        return neighbour cells of cell i
        :param i: cell index
        :return: neighbour cell list
        """
        return self.cell_neighbour[i]

    def is_cell_on_boundary(self, i, bd=None):
        """
        check if cell i is on boundary bd
        :param i: cell index, 0<=i<num_cell
        :param bd: boundary name, byte str
        :return: True or False
        """
        if i < 0 or i >= self.num_cell:
            return False
        if bd is not None:
            try:
                bid = self.boundary[bd].id
            except KeyError:
                return False
        for n in self.cell_neighbour[i]:
            if bd is None and n < 0:
                return True
            elif bd and n == bid:
                return True
        return False

    def is_face_on_boundary(self, i, bd=None):
        """
        check if face i is on boundary bd
        :param i: face index, 0<=i<num_face
        :param bd: boundary name, byte str
        :return: True or False
        """
        if i < 0 or i >= self.num_face:
            return False
        if bd is None:
            if self.neighbour[i] < 0:
                return True
            return False
        try:
            bid = self.boundary[bd].id
        except KeyError:
            return False
        if self.neighbour[i] == bid:
            return True
        return False

    def boundary_cells(self, bd):
        """
        return cell id list on boundary bd
        :param bd: boundary name, byte str
        :return: cell id generator
        """
        try:
            b = self.boundary[bd]
            return (self.owner[f] for f in range(b.start, b.start+b.num))
        except KeyError:
            return ()

    def _set_boundary_faces(self):
        """
        set faces' boundary id which on boundary
        :return: none
        """
        self.neighbour.extend([-10]*(self.num_face - self.num_inner_face))
        for b in self.boundary.values():
            self.neighbour[b.start:b.start+b.num] = [b.id]*b.num

    def _construct_cells(self):
        """
        construct cell faces, cell neighbours
        :return: none
        """
        cell_num = max(self.owner) + 1
        self.cell_faces = [[] for i in range(cell_num)]
        self.cell_neighbour = [[] for i in range(cell_num)]
        for i, n in enumerate(self.owner):
            self.cell_faces[n].append(i)
        for i, n in enumerate(self.neighbour):
            if n >= 0:
                self.cell_faces[n].append(i)
                self.cell_neighbour[n].append(self.owner[i])
            self.cell_neighbour[self.owner[i]].append(n)

    def _parse_mesh_data(self, path):
        """
        parse mesh data from mesh files
        :param path: path of mesh files
        :return: none
        """
        self.boundary = self.parse_mesh_file(os.path.join(path, 'boundary'), self.parse_boundary_content)
        self.points = self.parse_mesh_file(os.path.join(path, 'points'), self.parse_points_content)
        self.faces = self.parse_mesh_file(os.path.join(path, 'faces'), self.parse_faces_content)
        self.owner = self.parse_mesh_file(os.path.join(path, 'owner'), self.parse_owner_neighbour_content)
        self.neighbour = self.parse_mesh_file(os.path.join(path, 'neighbour'), self.parse_owner_neighbour_content)

    @classmethod
    def parse_mesh_file(cls, fn, parser):
        """
        parse mesh file
        :param fn: boundary file name
        :param parser: parser of the mesh
        :return: mesh data
        """
        try:
            with open(fn, "rb") as f:
                content = f.readlines()
                return parser(content, is_binary_format(content))
        except FileNotFoundError:
            print('file not found: %s'%fn)
            return None

    @classmethod
    def parse_points_content(cls, content, is_binary, skip=10):
        """
        parse points from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: points coordinates as numpy.array
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = np.array([ln[1:-2].split() for ln in content[n + 2:n + 2 + num]], dtype=float)
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    vv = np.array(struct.unpack('{}d'.format(num*3),
                                                buf[disp:num*3*struct.calcsize('d') + disp]))
                    data = vv.reshape((num, 3))
                return data
            n += 1
        return None


    @classmethod
    def parse_owner_neighbour_content(cls, content, is_binary, skip=10):
        """
        parse owner or neighbour from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: indexes as list
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = [int(ln) for ln in content[n + 2:n + 2 + num]]
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    data = struct.unpack('{}i'.format(num),
                                         buf[disp:num*struct.calcsize('i') + disp])
                return list(data)
            n += 1
        return None

    @classmethod
    def parse_faces_content(cls, content, is_binary, skip=10):
        """
        parse faces from content
        :param content: file contents
        :param is_binary: binary format or not
        :param skip: skip lines
        :return: faces as list
        """
        n = skip
        while n < len(content):
            lc = content[n]
            if is_integer(lc):
                num = int(lc)
                if not is_binary:
                    data = [[int(s) for s in ln[2:-2].split()] for ln in content[n + 2:n + 2 + num]]
                else:
                    buf = b''.join(content[n+1:])
                    disp = struct.calcsize('c')
                    idx = struct.unpack('{}i'.format(num), buf[disp:num*struct.calcsize('i') + disp])
                    disp = 3*struct.calcsize('c') + 2*struct.calcsize('i')
                    pp = struct.unpack('{}i'.format(idx[-1]),
                                       buf[disp+num*struct.calcsize('i'):
                                           disp+(num+idx[-1])*struct.calcsize('i')])
                    data = []
                    for i in range(num - 1):
                        data.append(pp[idx[i]:idx[i+1]])
                return data
            n += 1
        return None

    @classmethod
    def parse_boundary_content(cls, content, is_binary=None, skip=10):
        """
        parse boundary from content
        :param content: file contents
        :param is_binary: binary format or not, not used
        :param skip: skip lines
        :return: boundary dict
        """
        bd = {}
        num_boundary = 0
        n = skip
        bid = 0
        in_boundary_field = False
        in_patch_field = False
        current_patch = b''
        current_type = b''
        current_nFaces = 0
        current_start = 0
        while True:
            if n > len(content):
                if in_boundary_field:
                    print('error, boundaryField not end with )')
                break
            lc = content[n]
            if not in_boundary_field:
                if is_integer(lc.strip()):
                    num_boundary = int(lc.strip())
                    in_boundary_field = True
                    if content[n + 1].startswith(b'('):
                        n += 2
                        continue
                    elif content[n + 1].strip() == b'' and content[n + 2].startswith(b'('):
                        n += 3
                        continue
                    else:
                        print('no ( after boundary number')
                        break
            if in_boundary_field:
                if lc.startswith(b')'):
                    break
                if in_patch_field:
                    if lc.strip() == b'}':
                        in_patch_field = False
                        bd[current_patch] = Boundary(current_type, current_nFaces, current_start, -10-bid)
                        bid += 1
                        current_patch = b''
                    elif b'nFaces' in lc:
                        current_nFaces = int(lc.split()[1][:-1])
                    elif b'startFace' in lc:
                        current_start = int(lc.split()[1][:-1])
                    elif b'type' in lc:
                        current_type = lc.split()[1][:-1]
                else:
                    if lc.strip() == b'':
                        n += 1
                        continue
                    current_patch = lc.strip()
                    if content[n + 1].strip() == b'{':
                        n += 2
                    elif content[n + 1].strip() == b'' and content[n + 2].strip() == b'{':
                        n += 3
                    else:
                        print('no { after boundary patch')
                        break
                    in_patch_field = True
                    continue
            n += 1

        return bd
