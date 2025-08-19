# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created May 8th, 2025

@author: alfoa
"""
#External Modules--------------------begin
import os
import numpy as np
import time
import re
import pathlib
from typing import Dict, Tuple,  List, Union, Iterable
#External Modules--------------------end

#Internal Modules--------------------begin
#Internal Modules--------------------end

# ----------  internal helpers -------------------------------------------------
_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_SCALAR = re.compile(rf"^\s*([A-Za-z0-9_()]+)\s+({_FLOAT})\s*;")
_VECTOR = re.compile(rf"^\s*([A-Za-z0-9_()]+)\s+\(\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*\)\s*;")
_TIME_DIR = re.compile(r"^-?\d*\.?\d+(?:[eE][-+]?\d+)?$")

_AVAILABLE_FILE_TYPES = ['functionObjectProperties', 'volScalarField', 'volVectorField','surfaceScalarField', 'surfaceVectorField']

def checkAccessAndWaitIfStillAccessed(filename: str | pathlib.Path, gracePeriod: float = 1., timeout: float =100.):
  """
    This utility method is aimed to check the timestamp of last access of the file
    and wait (up to timeout+gracePeriod) if it is within the gracePeriod
    @ In, filename, str, the file name that needs to be checked
    @ In, gracePeriod, float, optional, the grace period (From last access)
    @ In, timeout, float, optional, the timeout (maximum time to wait till exiting)
    @ Out, ready, bool, ready to be accessed?
  """
  ready = True
  currentTime = time.time()
  lastAccessTime = os.path.getatime(filename)
  if (currentTime - lastAccessTime) < gracePeriod:
    ready = False
    t = time.time()
    while True:
      time.sleep(gracePeriod/100.)
      currentTime = time.time()
      lastAccessTime = os.path.getatime(filename)
      if (currentTime - lastAccessTime) >= gracePeriod:
        ready = True
        break
      elif currentTime - t > gracePeriod + timeout:
        break
  return ready


class openfoamOutputParser(object):
  """
    Class to parse different openFOAM output files
  """
  def __init__(self,
               caseDirectory: str | pathlib.Path,
               caseFileName: str | pathlib.Path,
               variables: List[str] = None, writeCentroids: bool = False,
               checkAccessAndWait: bool = False):
    """
     Constructor
     @ In, caseDirectory, str or Path, the case directory containing the time stamps and results
     @ In, caseFileName, str or Path, the case file name (.foam)
     @ In, variables, list, optional, list of the variables to retrieve. If None, all the variables
                                      that are supported by this parser (see _AVAILABLE_FILE_TYPES)
                                      will be retrieved. If the list of variables contain a file type (class)

     @ In, checkAccessAndWait, bool, optional, check the access time of the outputfiles and wait if too recent? Default= False
     @ Out, None
    """
    # check if pyvista is available, raise error otherwise
    try:
      pyvista = __import__("pyvista")
    except ImportError:
      raise ImportError("python library 'pyvista' not found and OpenFOAM Interface has been invoked. "
                        "Install pyvista through pip/conda (conda-forge) or invoke RAVEN installation procedure "
                        "(i.e. establish_conda_env.sh script) with the additional command line option "
                        "'--code-interface-deps'. See User Manual for additiona details!")
    # pointer to pyvista library
    self._pyvista = pyvista
    # list of variables to collect
    self._variables = variables
    # bool. Should the parser write in a dedicated csv file the centroids of the mesh?
    self._writeCentroids = writeCentroids
    # name of the case directory (directory containing all the input and output folders/files)
    self._caseDirectory = caseDirectory
    # case file name (OpenFOAM case file name)
    self._caseFileName = caseFileName
    # bool. check the access time of the outputfiles and wait if too recent?
    self._checkAccessAndWait = checkAccessAndWait

  def processOutputs(self):
    """
      Method to process OpenFOAM output files
      The results are returned in the results dictionary
      @ In, None
      @ Out, results, dict, dictionary of results ({'key':np.array}
    """
    variablesFound = []
    results = {}
    # check the field variables (outputs) that have been generated
    producedFieldVariables = self.checkFieldVariables()
    # if we need to check the access time we check it below
    ready = True
    if self._checkAccessAndWait:
      ready = checkAccessAndWaitIfStillAccessed(self._caseDirectory)
    if not ready:
      raise ImportError(f'ERROR: OpenFOAM Interface | CASE DIRECTORY "{self._caseDirectory}" NOT READY TO BE READ!!!!')
    # we read the uniform folder (functionObjectProperties file (with user applied functions) and cumulativeContErr) if they exists
    results['time'], data = self.uniformFolderAggregate(self._caseDirectory)
    variablesFound.extend(data.keys())
    newData = self._expandVariablesFromVectorToScalar(data)
    if self._variables is not None:
      newData = {k: v for k, v in newData.items() if any(k.strip().startswith(p.strip()) for p in self._variables)}
    variablesFound.extend(newData.keys())
    variablesFound = list(set(variablesFound))
    results.update(newData)
    # now we read the mesh data
    if self._variables is not None:
      fieldVars = set([el.split("|")[0] for el in self._variables])
    else:
      fieldVars = producedFieldVariables
    for i, var in enumerate(fieldVars):
      if var not in producedFieldVariables:
        continue
      foamFile = pathlib.Path(self._caseDirectory).expanduser().resolve() / pathlib.Path(self._caseFileName)
      _, v, c = self.aggregateFieldNumpy(field=var, foamCasefile=foamFile)
      if v is not None:
        expanded = self._expandVariablesFromVectorToScalar({var: v})
        results.update(expanded)
        variablesFound.extend(list(expanded.keys()) +[var])
        if self._writeCentroids and i == 0:
          np.savetxt(pathlib.Path(self._caseDirectory).parent / "centroids.csv",
                     np.c_[np.arange(c.shape[0]), c],   # prepend cell index
                     delimiter=",",
                     header="cell,x,y,z")
      # else not found, so the following check will fail and we will return the missing variables
    if self._variables is not None and len(set(self._variables) - set(variablesFound)) > 0:
      raise RuntimeError(f" The variables {set(self._variables) - set(variablesFound)} "
                         "have not been found in OpenFOAM output")
    return results

  def checkFieldVariables(self):
    """
      Method to scan the directories and retrieve the variable names (field and not)
      that have been produced by the simulation
      @ In, None
      @ Out, variables, list, list of variables
    """
    ##### helper functions inside the checkFieldVariables method
    def initialFields(caseDir: pathlib.Path, startTime: str = "0") -> List[str]:
      """
        List all vol*Field files in the chosen start-time directory.
        @ In, caseDir, Path, the case directory
        @ In, startTime, str, the start time (the time to check)
        @ Out, fields, list, list of fields variables
      """
      dir0 = caseDir / startTime
      if not dir0.is_dir():
        raise FileNotFoundError(f"Start-time directory '{dir0}' not found")
      # skip option dirs
      ignore = {"fvPatchField", "polyMesh"}
      # retrieve fields
      fields = [f.name for f in dir0.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.name not in ignore]
      return sorted(fields)
    # now read the variables
    caseDir = pathlib.Path(self._caseDirectory).expanduser().resolve()
    cdict = caseDir / "system" / "controlDict"
    if not cdict.is_file():
      raise FileNotFoundError(f"'system/controlDict' not found in {caseDir}")
    timedirs = sorted((d for d in caseDir.iterdir() if d.is_dir() and _TIME_DIR.match(d.name)),
                      key=lambda d: float(d.name))
    lastTimeDir = timedirs[-1].name
    fields = initialFields(caseDir, lastTimeDir)
    return fields


  @staticmethod
  def _expandVariablesFromVectorToScalar(data):
    """
      Method to expand the variables from vector to scalar (using the nomenclature
      varName_x, varName_y, varName_z if size ==3 or varName_1, varName_2 etc if size != 3)
      @ In, data, dict, dictionary of data {key:numpy.array}
      @ Out, newData, dict, expanded dictionary
    """
    newData = {}
    for var in data:
      if len(data[var].shape) == 2:
        # we need to expand
        lenLastDim = data[var].shape[-1]
        appendix = "xyz" if lenLastDim == 3 else [str(el+1) for el in list(range(lenLastDim))]
        for i, el in enumerate(appendix):
          newData[ f"{var}|{el}"] = data[var][:, i].flatten()
      elif len(data[var].shape) == 3:
        # we need to expand
        lenLastDim = data[var].shape[-1]
        appendix = "xyz" if lenLastDim == 3 else [str(el+1) for el in list(range(lenLastDim))]
        secondLastDimension = data[var].shape[-2]
        secondLastAppendix = "xyz" if secondLastDimension == 3 else [str(el+1) for el in list(range(secondLastDimension))]
        for i, el in enumerate(appendix):
          for j, cid in enumerate(secondLastAppendix):
            newData[ f"{var}|{cid}|{el}"] = data[var][:, j, i].flatten()
      else:
        newData[var] = data[var]
    return newData

  def _readCumulativeContErr(self, path: str | pathlib.Path) -> Tuple[List[int], Union[float, list]]:
    """
      Method to read the cumulativeContErr file
      @ In, path, str | Path, Path to the OpenFOAM file (e.g. '0.1/uniform/cumulativeContErr').
      @ Out, dimensions, list[int], (seven-tuple in M L T Θ etc.)
      @ Out, value, float  |  list[float], float for a scalar field, list[float] of length 3 for a vector field
    """
    path = pathlib.Path(path).expanduser().resolve()
    dimRe   = re.compile(r"dimensions\s+\[([0-9\s\-]+)\]\s*;?")
    vecRe    = re.compile(rf"\(\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*\)")
    scalarRe = re.compile(rf"value\s+({_FLOAT})\s*;?")

    dimensions = None
    value: Union[float, list, None] = None

    with path.open() as fh:
      for line in fh:
        if dimensions is None:
          m = dimRe.search(line)
          if m:
            dimensions = [int(n) for n in m.group(1).split()]
            continue
        if value is None:
          # try vector first
          m = vecRe.search(line)
          if m:
            value = [float(m.group(i)) for i in range(1, 4)]
            continue
          # fall back to scalar
          m = scalarRe.search(line)
          if m:
            value = float(m.group(1))
            continue
        if dimensions is not None and value is not None:
          break
    if dimensions is None or value is None:
      raise RuntimeError(f"Could not find 'dimensions' or 'value' in {path}")
    return dimensions, value

  def _functionObjectPropertiesParseDict(self, path: pathlib.Path) -> Dict[str, object]:
    """
      Method to parse the function object properties dictionary
      @ In, path, pathlib.Path, the path to the dict
      @ Out, out, dict, dictionary of the functionObjectProperties file.
    """
    out: Dict[str, object] = {}
    with path.open() as fh:
      for line in fh:
        # Scalars
        m = _SCALAR.match(line)
        if m:
          key, val = m.group(1), float(m.group(2))
          # ignore headers such as 'version', 'format', 'class', etc.
          if key not in {"version", "format", "class", "location", "object"}:
            out[key] = val
          continue
        # Vectors
        m = _VECTOR.match(line)
        if m:
          key = m.group(1)
          vec = np.array([float(m.group(i)) for i in range(2, 5)])
          out[key] = vec
    return out

  def uniformFolderAggregate(self, caseDir: str | pathlib.Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Method to walk the case, read every time directory's functionObjectProperties and cumulativeContErr (if present)
    and collect the values into time-sorted NumPy arrays.

    @ In, caseDir, str or pathlib.Path, the path to the dict
    @ Out, data, dict, dictionary of the functionObjectProperties data.
                       Keys are the names found in the files (e.g. 'average(p)', 'max(U)').
                       Each value is:
                        (N,)  float array  for scalars
                        (N,3) float array  for vectors
                       aligned so that index i in every array corresponds to times[i].
    @ Out, times, np.array, array of time stamps
    """
    caseDir = pathlib.Path(caseDir).expanduser().resolve()
    times, store = [], {}

    # discover time directories
    for d in sorted(caseDir.iterdir(), key=lambda p: float(p.name) if _TIME_DIR.match(p.name) else np.inf):
      if not (d.is_dir() and _TIME_DIR.match(d.name)):
        continue
      parsed = {}
      cumulativeContErrPath =  d / "uniform" / "cumulativeContErr"
      if cumulativeContErrPath.is_file():
        _, parsed['cumulativeContErr'] = self._readCumulativeContErr(cumulativeContErrPath)

      fpath = d / "uniform" / "functionObjects" / "functionObjectProperties"
      if not fpath.is_file():
        continue
      t = float(d.name)
      parsed.update(self._functionObjectPropertiesParseDict(fpath))
      times.append(t)
      # accumulate
      for key, val in parsed.items():
        store.setdefault(key, []).append(val)
    # convert to ndarray
    times = np.asarray(times)
    data = {k: np.stack(v) for k, v in store.items()}     # scalars become (N,); vectors (N,3)
    return times, data

  def _buildFieldReader(self, foamfile: pathlib.Path) :
    """
      This method is aimed to build the pyvista reader based on POpenFOAMReader
      It return a reader that loads only the internal mesh, cell data only.
      @ In, foamfile, pathlib.Path, the path to the "case.foam" file
      @ Out, rdr, pyvista.POpenFOAMReader, the POpenFOAMReader reader
    """
    rdr = self._pyvista.POpenFOAMReader(str(foamfile))
    # skip the zero time since at 0, no field data is generaly created (if not a restart)
    rdr.skip_zero_time = True
    # keep the internal cells, skip boundary patches (faster, lighter)
    rdr.disable_all_patch_arrays()
    rdr.enable_patch_array("internalMesh")
    # we stay with cell data (no cell -> point interpolation needed here)
    rdr.cell_to_point_creation = False
    # (optional) reconstructed vs decomposed – auto-detect usually fine:
    # NB. I leave it here in case a more granular control is requested in the future
    # rdr.case_type = "reconstructed"   # or "decomposed"
    return rdr

  def _collect(self, foamfile: pathlib.Path,
               field: str
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
      Grab 'field' over every available time step.
      Returns (times, values, centroids)
      @ In, foamfile, pathlib.Path, path to the foam file (e.g. case.foam)
      @ In, field, str, the field name to retrieve (e.g. U, p, etc.)
      @ Out, times, list, list of time stamps
      @ Out, values, np.ndarray, the stacked values (n_time_steps,n_cells,3 (x,y,z))
      @ Out, centroids, np.ndarray, the centroids coordinates (n_cells,3 (x,y,z))
    """

    rdr = self._buildFieldReader(foamfile)
    times = np.asarray(rdr.time_values, dtype=float)
    if times.size == 0:
      raise RuntimeError(f"No time folders detected in {foamfile.parent}")

    values: List[np.ndarray] = []
    centroids = None
    for t in times:
      rdr.set_active_time_value(float(t))
      mb = rdr.read() # MultiBlock
      mesh = mb["internalMesh"] if isinstance(mb, self._pyvista.MultiBlock) else mb
      if field not in mesh.cell_data:
        raise KeyError(f"volField '{field}' absent at time {t:g}")
      data = np.asarray(mesh.cell_data[field]).copy()
      if centroids is None:
        centroids = mesh.cell_centers().points # (Nc, 3)
      values.append(data.copy())

    return times, np.stack(values), centroids

  def aggregateFieldNumpy(self, field: str = "U",
                foamCasefile: str | None = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
      Aggregate 'field' over every available time step.
      Returns (times, values, centroids)
      @ In, foamCasefile, pathlib.Path, path to the foam file (e.g. case.foam)
      @ In, field, str, the volField (field) name to retrieve (e.g. U, p, etc.)
      @ Out, times, list, list of time stamps
      @ Out, values, np.ndarray, the stacked values (n_time_steps,n_cells,3 (x,y,z))
      @ Out, centroids, np.ndarray, the centroids coordinates (n_cells,3 (x,y,z))
    """
    caseDir = pathlib.Path(self._caseDirectory).expanduser().resolve()
    foamfile = (
        pathlib.Path(foamCasefile)
          if foamCasefile is not None
          else caseDir / (caseDir.name + ".foam")
      )
    if not foamfile.is_file():
      with open(foamfile, 'w'):
        print(f"OPENFoam Interface: Placeholder '.foam' file not found: {foamfile}. It has been created!")
    try:
      # we try to find the field in the "internalMesh" first
      times, values, centroids = self._collect(foamfile, field)
    except KeyError:
      # we return None
      times, values, centroids = None, None, None
    return times, values, centroids

