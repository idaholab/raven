import sys,os
import __builtin__
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as ncDS

# add raven framework to path
#frameworkDir = os.path.expanduser('~/projects/raven/framework')
#sys.path.append(frameworkDir)

# go to raven for loading environment
#origDir = os.getcwd()
#os.chdir(frameworkDir)
# set up environment
#import Driver

# go back to original dir
#os.chdir(origDir)

# load cached xarray module
#from utils import CachedXArray as CXA
import CachedXArray as CXA
import cached_ndarray as CND

try:
  __builtin__.profile
except AttributeError:
  # profiler not preset, so pass through
  def profile(func): return func

class DataObject:
  """base class"""
  def __init__(self,in_vars,out_vars,dynamic=False):
    self.in_vars = in_vars
    self.out_vars = out_vars
    self.dynamic = dynamic
    self._data = None

  def add_realization(self,info_dict):
    pass

  def get_data(self):
    return self._data

#
#
#
#
class XarrDO(DataObject):
  # covers both of the cached methods, depending on if "prealloc" is True
  def __init__(self, in_vars, out_vars, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    DataObject.__init__(self,in_vars,out_vars,dynamic=dynamic)
    self.vars = self.in_vars + self.out_vars
    self.var_dims = var_dims #dictionary of dimensions used
    if prealloc:
      self._data = CXA.CachedDataset(cacheSize=cacheSize,prealloc=prealloc,entries=in_vars)
    else:
      self._data = CXA.CachedDataset(cacheSize=cacheSize)

  @profile
  def add_realization(self,real_dict):
    #real_dict is of form {var:DataArray, var:DataArray}
    rlz = xr.Dataset(data_vars = real_dict)
    self._data.append(rlz)

  @profile
  def asDataset(self):
    return self._data.asDataset()

  @profile
  def toNetCDF4(self,fname,**kwargs):
    self._data.asDataset().to_netcdf(fname,**kwargs)

#
#
#
#
#
class NDCached(DataObject):
  def __init__(self, in_vars, out_vars, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    DataObject.__init__(self,in_vars,out_vars,dynamic=dynamic)
    self.vars = self.in_vars + self.out_vars
    if self.dynamic:
      self._data = CND.cNDarray(width=len(self.vars),dtype=object)
    else:
      self._data = CND.cNDarray(width=len(self.vars))

  def add_realization(self,real_dict):
    if self.dynamic:
      self._data.append(np.asarray([list(real_dict[var] for var in self.vars)],dtype=object))
    else:
      self._data.append(np.asarray([list(real_dict[var] for var in self.vars)]))

  def asDataset(self):
    if type(self._data) != xr.Dataset:
      data = self._data.getData()
      arrs = {} #TODO how to get it all converted correctly...
      for v,var in enumerate(self.vars):
        if type(data[0,v]) == float:
          arrs[var] = xr.DataArray(data[:,v],
                                   dims=['sample'],
                                   coords={'sample':range(len(self._data))},
                                   name=var)
        elif type(data[0,v]) == xr.DataArray:
          #arrs[var] = xr.merge(dat.rename(str(i)) for i,dat in enumerate(data[:,v]))
          arrs[var] = xr.concat(data[:,v], pd.Index(range(len(data[:,v])), name='sample'))
          arrs[var].rename(var)
        else:
          raise IOError('Unrecognized data type for var "{}": "{}"'.format(var,type(data[0,v])))
      self._data = xr.Dataset(arrs)
    return self._data

  def toNetCDF4(self,fname,**kwargs):
    self.asDataset().to_netcdf(fname)

#
#
#
#
class PureLists(DataObject):
  def __init__(self, in_vars, out_vars, dynamic=False, var_dims=None,cacheSize=100,prealloc=False):
    DataObject.__init__(self,in_vars,out_vars,dynamic=dynamic)
    self.vars = self.in_vars + self.out_vars
    self._data = CND.listOfLists(width=len(self.vars))

  def add_realization(self,real_dict):
    self._data.append([list(real_dict[var] for var in self.vars)])

  def asDataset(self):
    if type(self._data) != xr.Dataset:
      data = self._data.getData()
      arrs = {}
      for v,var in enumerate(self.vars):
        column = list(data[i][v] for i in range(len(self._data)))
        if type(column[0]) == float:
          arrs[var] = xr.DataArray(np.asarray(column),
                                   dims=['sample'],
                                   coords={'sample':range(len(self._data))},
                                   name=var)
        elif type(column[0]) == xr.DataArray:
          arrs[var] = xr.concat(column,pd.Index(range(len(self._data)), name='sample'))
        else:
          raise IOError('Unrecognized data type for var "{}": "{}"'.format(var,type(data[0][v])))
      self._data = xr.Dataset(arrs)
    return self._data

  def toNetCDF4(self,fname,**kwargs):
    self.asDataset().to_netcdf(fname)

#
#
#
#
class NpPrealloc(DataObject):
  def __init__(self, in_vars, out_vars, expectedSamples):
    DataObject.__init__(self,in_vars,out_vars)
    self.vars = self.in_vars + self.out_vars
    self.expectedSamples = expectedSamples
    self._data = np.zeros([len(self.vars),expectedSamples])
    self.counter = 0

  def add_realization(self,real_dict):
    #real_dict is of form {var:val}
    self._data[:,self.counter] = list(real_dict[v] for v in self.vars)

  def asDataset(self):
    if type(self._data) == np.ndarray:
      arrs = dict((var,xr.DataArray(self._data[v],
                                    dims=['sample'],
                                    coords={'sample':range(self.expectedSamples)}))
                                    for v,var in enumerate(self.vars))
      self._data = xr.Dataset(arrs)
    return self._data

  def toNetCDF4(self,fname,**kwargs):
    self.asDataset().to_netcdf(fname)
#
#
#
#
class XarrPreAlloc(DataObject):
  # FULL preallocation
  def __init__(self, in_vars, out_vars, expectedSamples, dynamic=False, var_dims=None):
    DataObject.__init__(self,in_vars,out_vars,dynamic=dynamic)
    self.vars = self.in_vars + self.out_vars
    self.var_dims = var_dims #dictionary of dimensions used
    set_data = dict((v,xr.DataArray(np.zeros(expectedSamples),
                                             dims=['sample'],
                                             coords={'sample':range(expectedSamples)})) for v in in_vars)
    self._data = xr.Dataset(set_data)
    self.counter = 0

  def add_realization(self,real_dict):
    #real_dict is of form {var:val}
    for var,val in real_dict.items():
      self._data[var].loc[{'sample':self.counter}] = val
    self.counter += 1

    #print '*'*80
    #print 'TESTING'
    #print self._data
    #new_arrs = dict((var,xr.DataArray([val],dims=['sample'],coords={'sample':[self.counter]})) for var,val in real_dict.items())
    #new_arrs = dict((var,xr.DataArray(val,dims=[],coords={})) for var,val in real_dict.items())
    #new = xr.Dataset(data_vars = new_arrs)
    #print ''
    #print 'new:'
    #print new
    #print ''
    #print 'dbg:'
    #self._data[dict(sample=self.counter)] = new
    #print self._data[dict(sample=self.counter)] = new#.update(new,inplace=True)
    #print ''
    #print 'after:'
    #print self._data
    #import sys;sys.exit()
    #self._data[dict(sample=self.counter)].update(real_dict,inplace=True)

  def asDataset(self):
    return self._data

  def toNetCDF4(self,fname,**kwargs):
    self._data.to_netcdf(fname,**kwargs)


#
#
#
#
class XarrOnDisc(DataObject):
  def __init__(self,inv,outv,name):
    DataObject.__init__(self,inv,outv)
    self._data = name
    self.counter = 0
    self.rid = 'RAVENsampleCounter'
    try:
      os.system('rm '+self._data)
    except:
      pass
    data = self._load()
    data.createDimension(self.rid)
    data.createVariable(self.rid,'u8')
    for v in inv+outv:
      data.createVariable(v,'f8',(self.rid,))
    data.close() #FIXME

  def _load(self):
    try:
      return ncDS(self._data,'a')
    except IOError:
      return ncDS(self._data,'w',format='NETCDF4')

  def add_realization(self,real_dict):
    data = self._load()
    for var,val in real_dict.items():
      data.variables[var][self.counter] = val
    data.variables[self.rid][self.counter] = self.counter
    self.counter += 1
    #data.close() #FIXME

  def asDataset(self):
    return xr.open_dataset(self._data,drop_variables=[self.rid])

  def toNetCDF4(self,*args,**kwargs):
    pass #already on file!

