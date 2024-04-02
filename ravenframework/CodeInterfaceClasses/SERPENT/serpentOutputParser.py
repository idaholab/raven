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
Created Feb 9th, 2024

@author: alfoa
"""
#External Modules--------------------begin
import os
import numpy as np
#External Modules--------------------end

#Internal Modules--------------------begin
from ravenframework.utils import utils
#Internal Modules--------------------end


serpentOutputAvailableTypes = ['ResultsReader', 'DetectorReader',
                               'DepletionReader', 'DepmtxReader',
                               'MicroXSReader', 'HistoryReader']

outputExtensions = {'ResultsReader': '_res.m',
                    'DetectorReader': '_det[bu].m',
                    'DepletionReader': '_dep.m',
                    'DepmtxReader': '_depmtx_[mat]_[bu]_[ss].m',
                    'MicroXSReader': '_mdx[bu].m',
                    'HistoryReader': '_his[bu].m'}

def checkCompatibilityFileTypesAndInputFile(inputFile, fileTypesToRead):
  """
    This method is aimed to check the compatibility of the output files request
    and the input file that needs to generate such output files
    @ In, inputFile, Files.File, input file from RAVEN
    @ In, fileTypesToRead, list, list of file types to read and check
    @ Out, None
  """
  with open(inputFile.getAbsFile(), 'r') as oi:
    data = oi.read()
    if 'include ' in data:
      counts = data.count("include ")
      additionalText = ''
      lines = data.split('\n')
      cnt = 0
      for line in lines:
        if 'include' in line.strip():
          includeFileName = line.split('include')[-1].split("%")[0].replace('"', '')
          additionalText += '\n' + open(os.path.join(inputFile.getPath().strip(), includeFileName.strip())).read() + '\n'
          cnt += 1
          if cnt == counts:
            break

    data += additionalText
    if 'DepmtxReader' in  fileTypesToRead:
      if 'set depmtx' not in data:
        raise ValueError("DepmtxReader file type has been requested but no 'depmtx' flag has been set in the input file!")
      else:
        optionSet = data.split('set depmtx')[1].strip()
        if not optionSet.startswith('1'):
          raise ValueError("DepmtxReader file type has been requested but 'set depmtx' flag is not set to '1'!")
    if 'DetectorReader' in fileTypesToRead:
      if 'det ' not in data:
        raise ValueError("DetectorReader file type has been requested but no detectors ('det' card) have been specified in the input file (and/or include files)!")
    if 'DepletionReader' in fileTypesToRead:
      if 'dep ' not in data:
        raise ValueError("DepletionReader file type has been requested but the input file is not for a depletion calculation ('dep' card not found)!")
    if 'MicroXSReader' in fileTypesToRead:
      raise NotImplementedError("MicroXSReader file type not available yet!")
      #if 'set mdep' not in data:
      #  raise Exception("MicroXSReader file type has been requested but no 'mdep' flag has been set in the input file!")
    if 'HistoryReader' in fileTypesToRead:
      raise NotImplementedError("HistoryReader file type not available yet!")
      #if 'set his' not in data:
      #  raise Exception("HistoryReader file type has been requested but no 'his' flag has been set in the input file!")
      #else:
      #  optionSet = data.split('set his')[1].strip()
      #  if not optionSet.startswith('1'):
      #    raise Exception("HistoryReader file type has been requested but 'set his' flag is not set to '1'!")


class SerpentOutputParser(object):
  """
    Class to parse different serpent output files
  """
  def __init__(self, fileTypes, fileRootName, eol = None):
    """
     Constructor
     @ In, fileTypes, list-like, list of file types to process
     @ In, fileRootName, str, file root name (from which the file names
                              for the different file types are inferred)
     @ In, eol, dict, dict of EOL targets {targetID1:value1,targetID2:value2, etc.}
     @ Out, None
    """
    # import serpent tools
    try:
      st = __import__("serpentTools")
    except ImportError:
      raise ImportError("serpentTools not found and SERPENT Interface has been invoked. Install serpentTools through pip!")
    self._st = st
    self._fileTypes = fileTypes
    self._fileRootName =  fileRootName
    self._data = {}
    self._eol = eol

  def processOutputs(self):
    """
      Method to process output files (self._fileTypes)
      The results are stored in self._data
      @ In, None
      @ Out, None
    """
    # we read the res file first since additional info can be found
    # there as burn up step (if any), etc.
    results, nSteps = self._resultsReader()

    for ft in self._fileTypes:
      if ft == 'DetectorReader':
        results.update(self._detectorReader(nSteps))
      elif ft == 'DepletionReader':
        results.update(self._depletionReader())
      elif ft == 'DepmtxReader':
        results.update(self._depmtxReader(nSteps))
      #elif ft == 'HistoryReader':
      #  results.update(self._historyReader(nSteps))
      #elif ft == 'MicroXSReader':
      #  results.update(self._microXSReader(nSteps))

    return results

  def _resultsReader(self):
    """
      Method to read and process data from the Results (_res.m) File
      @ In, None
      @ Out, resultsResults, dict, the result container
      @ Out, nSteps, int, the number of burn up steps (0 if no burn)
    """
    resultsResults = {}
    res = self._st.read(f"{self._fileRootName}{outputExtensions['ResultsReader']}")
    buSteps = res.get('burnStep')
    nSteps = 1 if buSteps is None else len(buSteps)

    for k, v in res.resdata.items():
      for eix in range(v.shape[-1]):
        kk = f'{k}_{eix}' if v.shape[-1] > 1 else f'{k}'
        if nSteps:
          if len(v.shape) > 1:
            if v.shape[0] == nSteps:
              resultsResults[kk] = v[:, eix]  if nSteps and len(v.shape) > 1 else np.asarray(v[eix])
            else:
              # it is a quantity that does not have results for burn up step 0 (e.g. capture of poisons)
              resultsResults[kk] = np.asarray([0.0]+v[:, eix].tolist())
          else:
            resultsResults[kk] = np.asarray([0.0]+np.atleast_1d(v[eix]).tolist())
        else:
          resultsResults[kk] = np.asarray(v[eix])

      if 'keff' in k.lower() and k.lower() != 'anakeff':
        rhoSigma, rhoLogSigma = None,  None
        rho, rhoLog = (v[0] - 1) / v[0],  np.log(v[0])
        if v.shape[0] > 1:
          # we have sigma
          rhoSigma, rhoLogSigma = (v[1] / v[0]) * rho,  (v[1] / v[0]) * rhoLog
        resultsResults[f'{k.replace("Keff", "Reactivity")}_{0}'
                       if rhoSigma is not None
                       else f'{k.replace("Keff", "Reactivity")}'] = rho*1e5
        if rhoSigma is not None:
          resultsResults[f'{k.replace("Keff", "Reactivity")}_{1}'] = rhoSigma*1e5
        resultsResults[f'{k.replace("Keff", "ReactivityLog")}_{0}'
                       if rhoLogSigma is not None
                       else f'{k.replace("Keff", "ReactivityLog")}'] = rhoLog*1e5
        if rhoLogSigma is not None:
          resultsResults[f'{k.replace("Keff", "ReactivityLog")}_{1}'] = rhoLogSigma*1e5
    if nSteps > 1 and self._eol is not None:
      # create a new variable that tells us the time where the keff < 1
      for target in self._eol:
        value = self._eol[target]
        if target not in res.resdata:
          raise ValueError(f"Target {target} for EOL calcs is not in result data")
        targetValues = res.resdata[target][:,0]
        sorting = np.argsort(targetValues)
        endOfLife = np.interp(value,targetValues[sorting],res.resdata['burnDays'][:,0][sorting],left=min(res.resdata['burnDays'][:,0]),right=max(res.resdata['burnDays'][:,0]))
        resultsResults[f'EOL_{target}'] = np.asarray([endOfLife]*targetValues.size)

    return resultsResults, nSteps

  def _detectorReader(self, buSteps):
    """
      Method to read and process data from the Detector File
      @ In, buSteps, int, number of burn up steps
      @ Out, detectorResults, dict, the result container
    """
    detectorResults = {}
    for bu in range(buSteps):
      det = self._st.read(f"{self._fileRootName}{outputExtensions['DetectorReader']}".replace("[bu]", f"{bu}"))
      for detectorName, detectorContent in det.detectors.items():
        indeces = detectorContent.indexes
        if len(indeces) == 0:
          # scalar detector
          varName = detectorName
          if varName not in detectorResults:
            # create array if the variable is not in the container yet
            detectorResults[varName] = np.zeros(buSteps)
            detectorResults[f"{varName}_err"] = np.zeros(buSteps)
          detectorResults[varName] =  float(detectorContent.tallies)
          detectorResults[f"{varName}_err"] =  float(detectorContent.errors)
        else:
          # grid-based detector
          grids = {}
          for d, dim in enumerate(indeces):
            gridName = dim.replace("mesh", "").upper()
            grids[d] = detectorContent.grids[gridName][:, -1]
          iterator = np.nditer(detectorContent.tallies, flags=['multi_index'])
          while not iterator.finished:
            val = detectorContent.tallies[iterator.multi_index]
            valErr = detectorContent.errors[iterator.multi_index]
            varName = detectorName
            for d, dIdx in enumerate(iterator.multi_index):
              varName += f"_{indeces[d]}_{grids[d][dIdx]}"
            if varName not in detectorResults:
              # create array if the variable is not in the container yet
              detectorResults[varName] = np.zeros(buSteps)
              detectorResults[f"{varName}_err"] = np.zeros(buSteps)
            detectorResults[varName][bu] = val
            detectorResults[f"{varName}_err"][bu] = valErr
            iterator.iternext()
    return detectorResults

  def _depletionReader(self):
    """
      Method to read and process data from the Depletion File
      @ In, None
      @ Out, depletionResults, dict, the result container
    """
    depletionResults = {}
    dep = self._st.read(f"{self._fileRootName}{outputExtensions['DepletionReader']}")
    depletionResults[f"time_days"] = dep.days
    depletionResults[f"burnup"] = dep.burnup
    for mat in dep.materials:
      # burnup of this specific material
      depletionResults[f"{mat}_burnup"] = dep.materials[mat].burnup
      depletionResults[f"{mat}_volume"] = dep.materials[mat].volume
      for idx, name in  enumerate(dep.materials[mat].names):
        for quantity, dd in zip(["activity", "adens","decayHeat","ingTox","inhTox","mdens","photonProdRate"],
                                [dep.materials[mat].activity, dep.materials[mat].adens, dep.materials[mat].decayHeat,
                                 dep.materials[mat].ingTox, dep.materials[mat].inhTox, dep.materials[mat].mdens,
                                 dep.materials[mat].photonProdRate]):
          if dd is not None:
            depletionResults[f"{mat}_{name}_{quantity}"] = dd[idx, :]
    return depletionResults

  def _depmtxReader(self, buSteps):
    """
      Method to read and process data from the Depmtx File
      @ In, buSteps, int, number of burn up steps
      @ Out, depmtxResults, dict, the result container. The matrices are actually dumped in pickled files
    """
    import glob
    import pickle as pk

    depmtxResults = {}
    for bu in range(buSteps):
      # _depmtx_[mat]_[bu]_[ss].m
      fileName =  f"{self._fileRootName}{outputExtensions['BumatReader']}".replace("[bu]", f"{bu}")
      fileName.replace("[mat]", "*").replace("[ss]", "*")
      materialAndSubstepFilenames = list(glob.glob(fileName))
      for ff in materialAndSubstepFilenames:
        bum = self._st.read(ff)
        infoString = ff.split("_depmtx_")[-1].replace(".m", "")
        materialName, _, substep = infoString.split("_")
        varName = f"flx_{materialName}_{substep}"
        if varName not in depmtxResults:
          depmtxResults[varName] = np.zeros(buSteps)
        depmtxResults[varName][bu] = bum.flx
        varName = f"filename_depmtx_zai_{materialName}_{substep}"
        if varName not in depmtxResults:
          depmtxResults[varName] = np.zeros(buSteps, dtype=str)
        depmtxResults[varName][bu] = f"{self._fileRootName}depmtx_{materialName}_{bu}_{substep}_serialized.pk"
        # dump serialized matrices
        pk.dump((bum.zai, bum.depmtx), f"{self._fileRootName}depmtx_{materialName}_{bu}_{substep}_serialized.pk")
    return depmtxResults
