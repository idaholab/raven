# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared helpers for optimization plotters that render a subset of generations as
animation frames and/or standalone PNGs.

This centralizes logic that was previously duplicated across the animated
plotters (DiversityRadarPlot, ConstraintActivityTimelinePlot,
SamplingCoverageMapPlot, TradeoffSlicePlot, HypervolumeMoviePlot,
ObjectiveContourAnimationPlot, and the enhanced OptParallelCoordinatePlot path):

  * sampleGenerations       - evenly down-sample generations to a frame cap
  * selectFrameIndices      - choose which rendered frames to persist as PNGs
  * parseGenerationSelector - map an explicit <generations> selector onto indices
  * resolveGenerations      - pick the render set (explicit selector overrides cap)
  * frameIndicesToSave      - PNG frame positions given the render set

The <generations> selector lets a user inspect specific generations instead of,
or in addition to, the evenly-sampled overview. When it is supplied, the plotter
renders exactly those generations (bypassing the auto-sample cap); when it is
omitted, the historical even-sampling / cap behavior is unchanged.
"""

import math

import numpy as np

from ...utils import InputData, InputTypes


GENERATIONS_DESCR = r"""Optional list of specific generations (batch ids) to render, overriding the
      even-sampling cap. When omitted, the plotter shows an evenly-spaced overview limited by
      <max_frames> (and, for saved PNGs, <frames_max>). Provide any mix of comma-separated tokens:
      an explicit generation id (e.g. "5"), the keywords "first" or "last", or an inclusive
      range "start:stop" / "start:stop:step" over generation ids (e.g. "0:20:5"). Requested
      generations that are not present in the data raise an error. Use this to compare specific
      generations side-by-side without emitting hundreds of frames."""


def addGenerationSelectorSpec(spec):
  """
    Register the shared optional <generations> node on a plotter's input spec.
    @ In, spec, InputData.ParameterInput, the plotter's (partially built) input specification
    @ Out, spec, InputData.ParameterInput, the same spec with <generations> added
  """
  spec.addSub(InputData.parameterInputFactory('generations', contentType=InputTypes.StringListType,
      descr=GENERATIONS_DESCR))
  return spec


def parseGenerationSelectorNode(spec):
  """
    Extract the raw <generations> tokens from a parsed input spec, if present.
    @ In, spec, InputData.ParameterInput, the plotter's parsed input
    @ Out, tokens, list or None, the raw string tokens, or None when the node is absent/empty
  """
  node = spec.findFirst('generations')
  if node is None or not node.value:
    return None
  tokens = [str(tok).strip() for tok in node.value if tok is not None and str(tok).strip()]
  return tokens or None


def sampleGenerations(generations, limit):
  """
    Evenly down-sample a sorted list of generations to at most `limit` entries, always
    keeping the first and last generation.
    @ In, generations, list, sorted unique generation identifiers
    @ In, limit, int, maximum number of generations to keep
    @ Out, (gens, indices), tuple(list, list), the selected generations and their
             indices into `generations`
  """
  n = len(generations)
  if limit >= n:
    return list(generations), list(range(n))
  positions = np.linspace(0, n - 1, limit, dtype=int)
  selected = []
  for idx in positions:
    if idx not in selected:
      selected.append(idx)
  cursor = 0
  while len(selected) < limit and cursor < n:
    if cursor not in selected:
      selected.append(cursor)
    cursor += 1
  selected = sorted(set(selected))
  if len(selected) > limit:
    selected = selected[:limit - 1] + [n - 1]
  elif selected and selected[-1] != n - 1:
    selected[-1] = n - 1
  selected = sorted(set(selected))
  return [generations[i] for i in selected], selected


def selectFrameIndices(total, frameMax):
  """
    Choose which of `total` rendered frames to persist as standalone PNGs, evenly strided
    and always including the final frame.
    @ In, total, int, number of rendered frames
    @ In, frameMax, int, maximum number of PNG frames to save
    @ Out, indices, list, positions (into the rendered sequence) to save
  """
  if total <= 0 or frameMax <= 0:
    return []
  if total <= frameMax:
    return list(range(total))
  stride = int(math.ceil(total / float(frameMax)))
  indices = list(range(0, total, stride))
  if indices and indices[-1] != total - 1:
    if len(indices) >= frameMax:
      indices[-1] = total - 1
    else:
      indices.append(total - 1)
  return sorted(set(indices))


def parseGenerationSelector(tokens, generations):
  """
    Map explicit <generations> tokens onto indices into the sorted `generations` list.
    @ In, tokens, list, raw string tokens (integers, "first"/"last", or "a:b[:step]" ranges)
    @ In, generations, list, sorted unique generation identifiers (numeric)
    @ Out, indices, list, sorted unique indices into `generations`
    Raises ValueError if a requested generation is absent or a token is malformed.
  """
  if not generations:
    raise ValueError('no generations are available to select from')
  # generation ids are integer-valued floats (e.g. batchId); index them by their int value
  idByValue = {}
  for pos, gen in enumerate(generations):
    idByValue[int(round(float(gen)))] = pos
  ordered = sorted(idByValue)
  selected = set()

  def _lookup(value):
    key = int(round(float(value)))
    if key not in idByValue:
      raise ValueError(f'requested generation "{value}" is not present in the data '
                       f'(available range {ordered[0]}..{ordered[-1]})')
    return idByValue[key]

  for token in tokens:
    low = token.lower()
    if low == 'first':
      selected.add(0)
    elif low == 'last':
      selected.add(len(generations) - 1)
    elif ':' in token:
      parts = token.split(':')
      if len(parts) not in (2, 3):
        raise ValueError(f'malformed range "{token}"; expected "start:stop" or "start:stop:step"')
      try:
        start = int(round(float(parts[0])))
        stop = int(round(float(parts[1])))
        step = int(round(float(parts[2]))) if len(parts) == 3 and parts[2].strip() else 1
      except ValueError:
        raise ValueError(f'malformed range "{token}"; start/stop/step must be numeric')
      if step <= 0:
        raise ValueError(f'range step in "{token}" must be positive')
      if stop < start:
        start, stop = stop, start
      hit = False
      for value in range(start, stop + 1, step):
        if value in idByValue:
          selected.add(idByValue[value])
          hit = True
      if not hit:
        raise ValueError(f'range "{token}" matched no generations present in the data '
                         f'(available range {ordered[0]}..{ordered[-1]})')
    else:
      try:
        selected.add(_lookup(token))
      except ValueError:
        raise
  return sorted(selected)


def resolveGenerations(generations, explicitTokens, defaultCap, maxFrames):
  """
    Decide which generations to render. An explicit <generations> selector wins and is honored
    exactly; otherwise the generations are evenly sampled up to the frame cap.
    @ In, generations, list, sorted unique generation identifiers
    @ In, explicitTokens, list or None, raw <generations> tokens (None -> auto-sample)
    @ In, defaultCap, int, default cap used when <max_frames> is not supplied
    @ In, maxFrames, int or None, user-supplied <max_frames> cap (None -> defaultCap)
    @ Out, (gens, indices), tuple(list, list), selected generations and their indices
    Raises ValueError (from parseGenerationSelector) for a bad explicit selector.
  """
  if explicitTokens:
    indices = parseGenerationSelector(explicitTokens, generations)
    return [generations[i] for i in indices], indices
  cap = maxFrames if maxFrames is not None else min(len(generations), defaultCap)
  cap = max(1, min(cap, len(generations)))
  return sampleGenerations(generations, cap)


def frameIndicesToSave(numRendered, explicit, saveFrames, frameMax):
  """
    Choose which rendered frames become standalone PNGs. When the user explicitly selected
    generations, all of them are saved (the user already bounded the count); otherwise the
    rendered frames are strided down to <frames_max>.
    @ In, numRendered, int, number of rendered frames
    @ In, explicit, bool, whether an explicit <generations> selector was used
    @ In, saveFrames, bool, the plotter's <save_frames> flag
    @ In, frameMax, int, the plotter's <frames_max> cap
    @ Out, indices, list, positions (into the rendered sequence) to save as PNGs
  """
  if not saveFrames:
    return []
  if explicit:
    return list(range(numRendered))
  return selectFrameIndices(numRendered, frameMax)
