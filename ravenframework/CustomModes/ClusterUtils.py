# Copyright 2017 Battelle Energy Alliance, LLC
#
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
  Shared, framework-independent utilities for the cluster SimulationModes
  (Slurm, PBS, MPI legacy).

  NOTE: this module intentionally has NO imports from the ravenframework
  package so that it can be unit tested without a built RAVEN installation.
"""

import os
import re
import math
import string
import subprocess


def parseSlurmTasksPerNode(spec):
  """
    Parses the SLURM_TASKS_PER_NODE environment variable format into a flat
    list of per-node task counts.
    The format is a comma separated list of entries, where each entry is
    either "N" (N tasks on one node) or "N(xM)" (N tasks on each of M nodes).
    For example "36(x2),20" -> [36, 36, 20].
    @ In, spec, str, the SLURM_TASKS_PER_NODE-style specification
    @ Out, counts, list(int), one entry per node with the task count
  """
  counts = []
  if spec is None:
    return counts
  for entry in spec.split(","):
    entry = entry.strip()
    if not entry:
      continue
    match = re.match(r"^(\d+)(\(x(\d+)\))?$", entry)
    if match is None:
      raise ValueError(f'Unparsable SLURM_TASKS_PER_NODE entry "{entry}" in "{spec}"')
    tasks = int(match.group(1))
    repeat = int(match.group(3)) if match.group(3) is not None else 1
    counts.extend([tasks] * repeat)
  return counts


def slurmNodeListFromScontrol(nodeList=None, tasksPerNode=None, runner=subprocess.run):
  """
    Expands a Slurm node list (e.g. "node[01-03],gpu01") into one line per
    task using "scontrol show hostnames" and the SLURM_TASKS_PER_NODE
    specification. This is the fallback node-discovery mechanism used when
    "srun hostname" is unavailable or fails.
    @ In, nodeList, str, optional, node list (defaults to $SLURM_JOB_NODELIST)
    @ In, tasksPerNode, str, optional, tasks-per-node spec (defaults to
      $SLURM_TASKS_PER_NODE, then $SLURM_CPUS_ON_NODE, then 1 per node)
    @ In, runner, callable, optional, subprocess.run-compatible callable
      (injectable for testing)
    @ Out, lines, list(str), one hostname entry per task (no trailing newline)
  """
  if nodeList is None:
    nodeList = os.environ.get("SLURM_JOB_NODELIST")
  if nodeList is None:
    return []
  result = runner(["scontrol", "show", "hostnames", nodeList],
                  capture_output=True, text=True, timeout=60)
  if result.returncode != 0:
    raise RuntimeError(f'"scontrol show hostnames {nodeList}" failed with code '
                       f'{result.returncode}: {result.stderr}')
  hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
  if tasksPerNode is None:
    tasksPerNode = os.environ.get("SLURM_TASKS_PER_NODE")
  if tasksPerNode is not None:
    counts = parseSlurmTasksPerNode(tasksPerNode)
  else:
    cpusOnNode = os.environ.get("SLURM_CPUS_ON_NODE")
    perNode = int(cpusOnNode) if cpusOnNode is not None else 1
    counts = [perNode] * len(hosts)
  if len(counts) < len(hosts):
    # be forgiving: pad with the last known count
    counts = counts + [counts[-1] if counts else 1] * (len(hosts) - len(counts))
  lines = []
  for host, count in zip(hosts, counts):
    lines.extend([host] * count)
  return lines


def readNodeFile(path):
  """
    Reads a node file (one node name per line, one line per available
    task/processor) and returns the stripped, non-empty lines.
    @ In, path, str, the path of the node file
    @ Out, lines, list(str), the node names (one entry per processor)
  """
  with open(path, "r") as nodeFileObject:
    return [line.strip() for line in nodeFileObject if line.strip()]


def computeBatchSize(numProcessors, numMPI, requestedBatchSize):
  """
    Computes the batch size that fits in the given number of processors.
    The batch size is the number of processors divided by numMPI (processors
    per run); floor/max keep the numbers reasonable.
    @ In, numProcessors, int, number of available processors (node file lines)
    @ In, numMPI, int, number of MPI processes per run
    @ In, requestedBatchSize, int, the batch size requested in the input
    @ Out, (batchSize, changed), (int, bool), the usable batch size and whether
      it had to be reduced from the requested one
  """
  maxBatchsize = max(int(math.floor(numProcessors / numMPI)), 1)
  if maxBatchsize < requestedBatchSize:
    return maxBatchsize, True
  return requestedBatchSize, False


def writeNodeSubFiles(lines, batchSize, numMPI, workingDir, prefix="node_"):
  """
    Splits the node list so that numMPI processors are available per batch
    slot, writing one node file per slot (node_0, node_1, ...): the files
    referenced by the "%BASE_WORKING_DIR%/node_%INDEX%" placeholder command.
    @ In, lines, list(str), node names, one entry per processor
    @ In, batchSize, int, number of batch slots
    @ In, numMPI, int, number of MPI processes per run
    @ In, workingDir, str, directory in which to write the files
    @ In, prefix, str, optional, file name prefix
    @ Out, written, list(str), the paths written
  """
  written = []
  for i in range(batchSize):
    subFileName = os.path.join(workingDir, f"{prefix}{i}")
    with open(subFileName, "w") as subNodeFile:
      for line in lines[i*numMPI : (i+1)*numMPI]:
        subNodeFile.write(line.rstrip("\n") + "\n")
    written.append(subFileName)
  return written


def buildMPIPrecommand(mpiExec, mpiParams, nodeCommand, numMPI, existingPrecommand):
  """
    Creates the mpiexec precommand. With defaults the precommand is
    "mpiexec -f nodeFile -n numMPI <existing precommand>".
    @ In, mpiExec, str, the mpi executor (e.g. "mpiexec")
    @ In, mpiParams, list(str), extra parameters for mpi
    @ In, nodeCommand, str, the node-selection portion (e.g. "-f nodefile"),
      or " " when running on the local machine only
    @ In, numMPI, int, number of MPI processes per run
    @ In, existingPrecommand, str, the pre-existing precommand to append
    @ Out, precommand, str, the assembled precommand
  """
  mpiParamsStr = (" ".join(mpiParams) + " ") if mpiParams else ""
  return mpiExec + " " + mpiParamsStr + nodeCommand + " -n " + str(numMPI) + " " + existingPrecommand


def sanitizeJobName(jobName, maxLength=None):
  """
    Validates and (optionally) shortens a scheduler job name. Only
    alphanumeric characters, "_" and "-" are allowed. When maxLength is given
    and exceeded, the name is shortened keeping the head and the last 4
    characters (e.g. maxLength=15: first 10 + "-" + last 4).
    @ In, jobName, str, the requested job name
    @ In, maxLength, int, optional, maximum allowed length
    @ Out, jobName, str, the validated (possibly shortened) job name
  """
  validChars = set(string.ascii_letters) | set(string.digits) | set('-_')
  if any(char not in validChars for char in jobName):
    raise ValueError('JobName can only contain alphanumeric, "_" and "-" '
                     'characters! Received: ' + jobName)
  if maxLength is not None and len(jobName) > maxLength:
    jobName = jobName[:maxLength-5] + '-' + jobName[-4:]
  return jobName


def buildSrunPrecommand(numMPI, mpiParams, existingPrecommand):
  """
    Creates a Slurm-native "srun" precommand instead of the mpiexec+nodefile
    one. Slurm tracks per-step resource assignment itself, so no node files
    are needed: "--exact" gives each step exactly the requested tasks and
    "--overlap" allows the steps of a batch to share the allocation. Works
    with all major MPI stacks via PMI/PMIx.
    @ In, numMPI, int, number of MPI processes (tasks) per run
    @ In, mpiParams, list(str), extra parameters passed to srun
    @ In, existingPrecommand, str, the pre-existing precommand to append
    @ Out, precommand, str, the assembled srun precommand
  """
  mpiParamsStr = (" ".join(mpiParams) + " ") if mpiParams else ""
  return "srun --overlap --exact -n " + str(numMPI) + " " + mpiParamsStr + existingPrecommand
