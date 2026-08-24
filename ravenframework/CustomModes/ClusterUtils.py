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
