"""
    The Data module contains some example classes which define the Data structure.
"""
from __future__ import print_function, division
import numpy as np
import copy

class GenericData():
    """
    The generic Data class which all user Data classes must
    inherit from.
    It contains functions to save/read/copy and to do mathematical
    operations used in the integration routines.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        self.coords = [c for c in coords]
        self.load(file=file,data=data)
        self.user_init(coords=coords,file=file,data=data)

    def user_init(self,coords=(0,0),file=None,data=None):
        return
    def load(self,file=None,data=None):
        """
        Load any data or set it from the class function

        Parameters
        ----------
        file : hdf5 group object
            If not None then this contains the data columns
            which we want to read.
        data : class
            If not None then this is a class which contains
            the data columns we want to copy.
        """
        if data is not None:
            for c in self.data_cols:
                setattr(self,c,getattr(data,c))
        elif file is None:
            for c in self.data_cols:
                setattr(self,c,self.func())
        else:
            for c in self.data_cols:
                setattr(self,c,file[c][...])
        self.user_load(file=file,data=data)
    def user_load(self,file=None,data=None):
        return
    def func(self):
        """Function which sets the data value"""
        self.value = 0
    def copy(self):
        """Copy function."""
        import copy
        return copy.copy(self)
    def save(self,file):
        """
        Save the contents to an hdf5 file.

        Parameters
        ----------
        file : hdf5 group
            The hdf5 container which we want to save in.
        """
        #grp = file.create_group('Data')
        for c in self.data_cols:
            file.create_dataset(c,data=getattr(self,c))
        self.user_save(file)
    def user_save(self,file):
        return
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value

    def __eq__(self,d2):
        return vars(self) == vars(d2)
    # Below are functions to handle addition/subtraction/multiplication/division
    def __rmul__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            try:
                setattr(newdat,d,getattr(newdat,d)*getattr(val,d))
            except AttributeError:
                setattr(newdat,d,getattr(newdat,d)*val)
        return newdat
    def __mul__(self,val):
        return self.__rmul__(val)
    def __radd__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            setattr(newdat,d,getattr(newdat,d) + getattr(val,d))
        return newdat
    def __add__(self,val):
        return self.__radd__(val)
    def __rsub__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            setattr(newdat,d,getattr(val,d)-getattr(newdat,d))

        return newdat
    def __sub__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            setattr(newdat,d,getattr(newdat,d) - getattr(val,d))
        return newdat
    def __rtruediv__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            try:
                setattr(newdat,d,getattr(val,d)/getattr(newdat,d))
            except AttributeError:
                 setattr(newdat,d,val/getattr(newdat,d))

        return newdat
    def __truediv__(self,val):
        newdat = copy.deepcopy(self)
        for d in newdat.data_cols:
            try:
                setattr(newdat,d,getattr(newdat,d)/getattr(val,d))
            except AttributeError:
                 setattr(newdat,d,getattr(newdat,d)/val)
        return newdat

class Empty(GenericData):
    """
    Simple Data class which does nothing.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)
        self.value = 0

class SimpleTest1D(GenericData):
    """
    1D test class of a gaussian.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)
    def func(self):
        """Function which sets the data value"""
        xc = self.coords[0]
        cx = 0
        cy = 0
        s = .1
        res = np.exp(-((xc-cx)**2)/(2*s**2))

        return res
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value

class CircleTest2D(GenericData):
    """
    2D test class which consists of a central gaussians.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)
    def func(self):
        """Function which sets the data value"""
        xc,yc = self.coords
        cx = 0
        cy = 0
        s = .3
        res = np.exp(-((xc-cx)**2+(yc-cy)**2)/(2*s**2))

        return res
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value

class SimpleTest2D(GenericData):
    """
    2D test class which consists of two gaussians.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)
    def func(self):
        """Function which sets the data value"""
        xc,yc = self.coords
        cx = .65 + .5* 2.**(-8)
        cy = .65 +  .5* 2.**(-8)
        s = .1
        res = np.exp(-((xc-cx)**2+(yc-cy)**2)/(2*s**2))
        cx = .3 + .5* 2.**(-8)
        cy = .3 + .5* 2.**(-8)
        s = .1
        res += np.exp(-((xc-cx)**2+(yc-cy)**2)/(2*s**2))
        return res
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value

class SpiralTest2D(GenericData):
    """
    2D test class which consists of a one-armed spiral.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)

    def func(self):
        """Function which sets the data value"""
        xc,yc = self.coords
        r = np.sqrt( xc**2 + yc**2)
        p = np.arctan2(yc,xc)

        ps = np.log(r/1)/.2
        xs = r*np.cos(ps)
        ys = r*np.sin(ps)
        res = np.exp(-((xc-xs)**2 + (yc-ys)**2)/(2*.3**2))
        if np.isnan(res) or np.isinf(res):
            res = 1
        return res
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value


class SpiralTest3D(GenericData):
    """
    3D test class which consists of a one-armed spiral that follows a
    gaussian distrubtion in the vertical direction.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)

    def func(self):
        """Function which sets the data value"""
        xc,yc,zc = self.coords
        r = np.sqrt( xc**2 + yc**2)
        p = np.arctan2(yc,xc)

        ps = np.log(r/1)/.2
        xs = r*np.cos(ps)
        ys = r*np.sin(ps)
        res = np.exp(-((xc-xs)**2 + (yc-ys)**2)/(2*.3**2))
        if np.isnan(res) or np.isinf(res):
            res = 1
        return res * np.exp(-zc**2/(2*.4**2))
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value
