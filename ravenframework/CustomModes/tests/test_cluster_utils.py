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
  Unit tests for CustomModes/ClusterUtils.py (node discovery, batch sizing,
  node-file splitting, precommand construction, job-name sanitation).

  These tests deliberately load ClusterUtils directly from its file path so
  they can run WITHOUT a built RAVEN installation (no Crow, no framework
  imports):

    python3 ravenframework/CustomModes/tests/test_cluster_utils.py
"""

import importlib.util
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.abspath(os.path.join(HERE, os.pardir, "ClusterUtils.py"))
spec = importlib.util.spec_from_file_location("ClusterUtils", MODULE_PATH)
ClusterUtils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ClusterUtils)


class FakeCompletedProcess:
  """ Minimal stand-in for subprocess.CompletedProcess """
  def __init__(self, returncode=0, stdout="", stderr=""):
    self.returncode = returncode
    self.stdout = stdout
    self.stderr = stderr


class TestParseSlurmTasksPerNode(unittest.TestCase):
  """ Tests for parseSlurmTasksPerNode """

  def testSimple(self):
    self.assertEqual(ClusterUtils.parseSlurmTasksPerNode("4"), [4])

  def testRepeated(self):
    self.assertEqual(ClusterUtils.parseSlurmTasksPerNode("36(x2),20"), [36, 36, 20])

  def testMixed(self):
    self.assertEqual(ClusterUtils.parseSlurmTasksPerNode("2,3(x3),1"), [2, 3, 3, 3, 1])

  def testNone(self):
    self.assertEqual(ClusterUtils.parseSlurmTasksPerNode(None), [])

  def testInvalid(self):
    with self.assertRaises(ValueError):
      ClusterUtils.parseSlurmTasksPerNode("bogus(x)")


class TestSlurmNodeListFromScontrol(unittest.TestCase):
  """ Tests for slurmNodeListFromScontrol with an injected fake scontrol """

  def testExpansion(self):
    def fakeRunner(cmd, capture_output, text, timeout):
      self.assertEqual(cmd, ["scontrol", "show", "hostnames", "node[01-02]"])
      return FakeCompletedProcess(stdout="node01\nnode02\n")
    lines = ClusterUtils.slurmNodeListFromScontrol(nodeList="node[01-02]",
                                                   tasksPerNode="2(x2)",
                                                   runner=fakeRunner)
    self.assertEqual(lines, ["node01", "node01", "node02", "node02"])

  def testFailureRaises(self):
    def fakeRunner(cmd, capture_output, text, timeout):
      return FakeCompletedProcess(returncode=1, stderr="boom")
    with self.assertRaises(RuntimeError):
      ClusterUtils.slurmNodeListFromScontrol(nodeList="nodeXX",
                                             tasksPerNode="1",
                                             runner=fakeRunner)

  def testNoNodeList(self):
    # no node list available at all -> empty result, no crash
    old = os.environ.pop("SLURM_JOB_NODELIST", None)
    try:
      self.assertEqual(ClusterUtils.slurmNodeListFromScontrol(nodeList=None,
                                                              tasksPerNode="1",
                                                              runner=None), [])
    finally:
      if old is not None:
        os.environ["SLURM_JOB_NODELIST"] = old


class TestNodeFiles(unittest.TestCase):
  """ Tests for readNodeFile / writeNodeSubFiles """

  def setUp(self):
    self.workDir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.workDir, ignore_errors=True)

  def testReadNodeFileSkipsBlanks(self):
    path = os.path.join(self.workDir, "nodes")
    with open(path, "w") as f:
      f.write("a\n\nb\n  \nc\n")
    self.assertEqual(ClusterUtils.readNodeFile(path), ["a", "b", "c"])

  def testWriteNodeSubFiles(self):
    # 6 processors, numMPI=2, batchSize=3 -> node_0..node_2 with 2 lines each
    lines = ["n1", "n1", "n2", "n2", "n3", "n3"]
    written = ClusterUtils.writeNodeSubFiles(lines, 3, 2, self.workDir)
    self.assertEqual([os.path.basename(p) for p in written],
                     ["node_0", "node_1", "node_2"])
    self.assertEqual(ClusterUtils.readNodeFile(written[0]), ["n1", "n1"])
    self.assertEqual(ClusterUtils.readNodeFile(written[1]), ["n2", "n2"])
    self.assertEqual(ClusterUtils.readNodeFile(written[2]), ["n3", "n3"])


class TestComputeBatchSize(unittest.TestCase):
  """ Tests for computeBatchSize """

  def testFits(self):
    self.assertEqual(ClusterUtils.computeBatchSize(8, 2, 4), (4, False))

  def testClamped(self):
    self.assertEqual(ClusterUtils.computeBatchSize(8, 2, 10), (4, True))

  def testAtLeastOne(self):
    self.assertEqual(ClusterUtils.computeBatchSize(1, 4, 2), (1, True))

  def testExactFit(self):
    self.assertEqual(ClusterUtils.computeBatchSize(4, 4, 1), (1, False))


class TestBuildMPIPrecommand(unittest.TestCase):
  """ Tests for buildMPIPrecommand (matches the legacy string construction) """

  def testDefault(self):
    pre = ClusterUtils.buildMPIPrecommand("mpiexec", [], "-f /wd/nodes", 4, "")
    self.assertEqual(pre, "mpiexec -f /wd/nodes -n 4 ")

  def testWithParamsAndExisting(self):
    pre = ClusterUtils.buildMPIPrecommand("mpiexec", ["--bind-to core"],
                                          "-f %BASE_WORKING_DIR%/node_%INDEX% ",
                                          2, "oldpre")
    self.assertEqual(pre, "mpiexec --bind-to core -f %BASE_WORKING_DIR%/node_%INDEX%  -n 2 oldpre")

  def testLocalMachine(self):
    # not in a cluster: nodeCommand is a single space (legacy behavior)
    pre = ClusterUtils.buildMPIPrecommand("mpiexec", [], " ", 2, "")
    self.assertEqual(pre, "mpiexec   -n 2 ")


class TestSanitizeJobName(unittest.TestCase):
  """ Tests for sanitizeJobName """

  def testValid(self):
    self.assertEqual(ClusterUtils.sanitizeJobName("my_job-1"), "my_job-1")

  def testInvalidRaises(self):
    with self.assertRaises(ValueError):
      ClusterUtils.sanitizeJobName("bad name!")

  def testTruncation(self):
    # PBS behavior: 15-char limit -> first 10 + '-' + last 4
    name = "abcdefghijklmnopqrst"
    self.assertEqual(ClusterUtils.sanitizeJobName(name, maxLength=15),
                     "abcdefghij-qrst")

  def testNoTruncationWhenShort(self):
    self.assertEqual(ClusterUtils.sanitizeJobName("short", maxLength=15), "short")


if __name__ == "__main__":
  unittest.main(verbosity=2)
