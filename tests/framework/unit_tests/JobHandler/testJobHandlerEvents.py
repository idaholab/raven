# Copyright 2026 Battelle Energy Alliance, LLC
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
  Unit tests for JobHandler per-job completion events.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import sys

ravenDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), os.pardir, os.pardir, os.pardir, os.pardir))
frameworkDir = os.path.join(ravenDir, 'framework')
sys.path.append(ravenDir)

from ravenframework import JobHandler
from ravenframework.utils import utils

utils.find_crow(frameworkDir)

results = {"pass": 0, "fail": 0}


def checkTrue(comment, value):
  """
    Check that a condition is true.
    @ In, comment, str, description of the check
    @ In, value, bool, condition value
    @ Out, None
  """
  if value:
    results["pass"] += 1
  else:
    print("checking answer", comment, "is not True")
    results["fail"] += 1


def checkEqual(comment, value, expected):
  """
    Check that two values are equal.
    @ In, comment, str, description of the check
    @ In, value, object, actual value
    @ In, expected, object, expected value
    @ Out, None
  """
  if value == expected:
    results["pass"] += 1
  else:
    print("checking answer", comment, value, "!=", expected)
    results["fail"] += 1


class FinishedRunner:
  """
    Minimal runner that becomes done immediately after start.
  """
  def __init__(self, identifier):
    """
      Constructor.
      @ In, identifier, str, runner identifier
      @ Out, None
    """
    self.identifier = identifier
    self.uniqueHandler = "any"
    self.groupId = None
    self.clientRunner = False
    self.args = []
    self.started = False
    self.times = []

  def start(self):
    """
      Mark the runner as started.
      @ In, None
      @ Out, None
    """
    self.started = True

  def isDone(self):
    """
      Report completion status.
      @ In, None
      @ Out, bool, True once started
    """
    return self.started

  def trackTime(self, event):
    """
      Record timing labels requested by JobHandler.
      @ In, event, str, timing event
      @ Out, None
    """
    self.times.append(event)

  def getReturnCode(self):
    """
      Return successful execution.
      @ In, None
      @ Out, int, zero return code
    """
    return 0

  def getMetadata(self):
    """
      Return no metadata.
      @ In, None
      @ Out, None
    """
    return None

  def kill(self):
    """
      Runner kill hook.
      @ In, None
      @ Out, None
    """
    pass


runInfo = {
  "maxQueueSize": 1,
  "batchSize": 1,
  "parallelMethod": None,
  "internalParallel": False,
  "Nodes": [],
}

handler = JobHandler.JobHandler()
handler.applyRunInfo(runInfo)
handler.initialize()

runner = FinishedRunner("job_event_test")
handler.reAddJob(runner)

event = handler.getJobEvent("job_event_test")
checkTrue("event created by reAddJob", event is not None)
checkTrue("event initially unset", not event.is_set())

handler.fillJobQueue()
checkTrue("runner started by fillJobQueue", runner.started)
checkTrue("event still unset while runner has not been cleaned", not event.is_set())

handler.cleanJobQueue()
checkTrue("event set by cleanJobQueue", event.is_set())
checkTrue("isThisJobFinished sees finished runner", handler.isThisJobFinished("job_event_test"))

finished = handler.getFinished(jobIdentifier="job_event_test")
checkEqual("one finished runner returned", len(finished), 1)
checkTrue("finished runner is original runner", finished[0] is runner)
checkTrue("event removed after getFinished cleanup", handler.getJobEvent("job_event_test") is None)

print(results)
sys.exit(results["fail"])
