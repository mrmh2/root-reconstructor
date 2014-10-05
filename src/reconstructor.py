import os
import re

import numpy as np
from skimage.io import use_plugin, imread

from coords2d import Coords2D

use_plugin('freeimage')

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def load_segmentation_maps(slice_dir):

    image_files = os.listdir(slice_dir)

    image_files = sorted_nicely(image_files)

    smaps = [SegmentationMap(os.path.join(slice_dir, im_file))
             for im_file in image_files]

    return smaps

def slice_from_same_cell(slice1, slice2):

    if slice1 is None or slice2 is None:
        return False

    area1 = slice1.pixel_area
    area2 = slice2.pixel_area

    dist = slice1.centroid.dist(slice2.centroid)

    area_ratio = float(area1) / area2

    if 0.5 < area_ratio < 1.5 and dist < 20:
        return True
    else:
        return False
        
def find_slice_links(slice_map1, slice_map2):
    matches = {}

    for cell_slice in slice_map1.cells.values():
        c = cell_slice.centroid
        candidate_slice = slice_map2.cell_at(c)
        if slice_from_same_cell(cell_slice, candidate_slice):
            #matches.append((cell_slice.ID, candidate_slice.ID))
            matches[cell_slice.ID] = candidate_slice.ID

    return matches

class CellSlice(object):

    def __init__(self, ID, coord_list):
        self.ID = ID
        self.coord_list = coord_list
        self.pixel_area = len(coord_list[0])
        self.x_coords = coord_list[0]
        self.y_coords = coord_list[1]

    @property
    def centroid(self):
        return Coords2D(sum(self.x_coords), 
                        sum(self.y_coords)) / self.pixel_area

    @property
    def summary(self):
        return "<ID: %d, pixel_area: %d, centroid: %s>" % (self.ID, 
                                                           self.pixel_area, 
                                                           self.centroid)

    def __repr__(self):
        return "<CellSlice, ID %d>" % self.ID

class SegmentationMap(object):

    def __init__(self, image_file):
        self.im_array = imread(image_file)
        self.internal_cc = None

    @property
    def cells(self):
        if self.internal_cc is not None:
            return self.internal_cc
        else:
            self.internal_cc = cell_dict_from_image_array(self.im_array)
            return self.internal_cc

    def cell_at(self, position):
        x, y = position

        ID = self.im_array[x, y]
        if ID == 0:
            return None

        return self.cells[self.im_array[x, y]]

    @property
    def all_ids(self):
        with_zero = list(np.unique(self.im_array))
        with_zero.remove(0)
        return with_zero

    def coord_list(self, cID):
        return np.where(self.im_array == cID)

class ReconstructedCell(object):

    def __init__(self, slice_dict):
        self.slice_dict = slice_dict

    def add_slice(self, layer, cellslice):
        self.slice_dict[layer] = cellslice

    @property
    def pixel_area(self):
        return sum(cellslice.pixel_area
                   for cellslice in self.slice_dict.values())

    @property
    def z_extent(self):
        return len(self.slice_dict.keys())

    def measure_total_intensity(self, idata):
        total_intensity = 0

        for sID, cellslice in self.slice_dict.items():
            #z_correction = 1 + (0.03 * sID)
            z_correction = 1
            total_intensity += z_correction * sum(idata[sID][cellslice.coord_list])

        return total_intensity

    def measure_mean_intensity(self, idata):
        total_intensity = self.measure_total_intensity(idata)
        area = self.pixel_area

        return float(total_intensity) / area

    def __repr__(self):
        return "<ReconstructedCell: %s>" % self.slice_dict.__repr__()

    def simple_string_rep(self):
        return ",".join("%d:%d" % (sID, cellslice.ID)
                        for sID, cellslice
                        in self.slice_dict.items())

def parse_recons_line(line):
    return [map(int, p.split(':')) for p in line.split(',')]

class Reconstruction(object):

    def __init__(self, smaps, start=0):
        self.smaps = smaps
        self.rcells = []
        self.lut = {}
        z = start
        for ID in smaps[z].all_ids:
            rcell = ReconstructedCell({z: self.smaps[z].cells[ID]})
            self.rcells.append(rcell)
            self.lut[(z, ID)] = rcell

    def extend(self, level):
        z = level
        matches = find_slice_links(self.smaps[z], self.smaps[z+1])

        for f, t in matches.iteritems():
            try:
                rcell = self.lut[(z, f)]
                rcell.add_slice(z+1, self.smaps[z+1].cells[t])
            except KeyError:
                rcell = ReconstructedCell({z+1: self.smaps[z+1].cells[t]})
                self.rcells.append(rcell)

            self.lut[(z+1, t)] = rcell

    def cells_larger_then(self, zext):
        return [rcell for rcell in self.rcells
                if rcell.z_extent >= zext]

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            f.write('\n'.join([rcell.simple_string_rep()
                               for rcell in self.rcells]))

    @classmethod
    def from_seg_Dir(cls, seg_dir, start=0):
        smaps = load_segmentation_maps(seg_dir)

        recons = cls(smaps, start)

        return recons
        
    @classmethod
    def from_file_and_dir(cls, filename, seg_dir):
        with open(filename) as f:
            lines = f.readlines()

        smaps = load_segmentation_maps(seg_dir)

        # prl = parse_recons_line("0:5,1:10,2:57,3:63,4:78,5:100")
        # print  {z: smaps[z].cells[ID] for z, ID in prl}

        # for l in lines:
        #     for reference in l.strip().split(','):
        #         layer, ID = reference.split(':')

        recons = cls(smaps)

        return recons

def cell_dict_from_image_array(i_array):

    cd = {cid: CellSlice(cid, np.where(i_array == cid))
          for cid in np.unique(i_array)}

    del(cd[0])

    return cd
