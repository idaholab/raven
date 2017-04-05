#!/usr/bin/env python
import os
import glob
import shutil
import unittest
import mooseutils
import time
import subprocess

class TestVectorPostprocessorReader(unittest.TestCase):
    """
    Test use of MooseDataFrame for loading/reloading csv files.
    """

    def setUp(self):
        """
        Define the test filename.
        """
        self._pattern = os.path.abspath('../../test_files/vpp_*.csv')

    def copyfiles(self, copytime=True):
        """
        Copy the saved file locally.
        """
        for f in glob.glob(self._pattern):
            if f.endswith('time.csv') and not copytime:
                continue
            shutil.copyfile(f, os.path.basename(f))

    def tearDown(self):
        """
        Remove local copy of files.
        """

        for f in glob.glob(self._pattern):
            fname = os.path.basename(f)
            if os.path.exists(fname):
                os.remove(fname)

    def testBasic(self):
        """
        Test that if a file exists it is loaded w/o error.
        """
        self.copyfiles()

        # Load the data
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertEqual(data.filename, os.path.basename(self._pattern))
        self.assertTrue(data._timedata)
        self.assertTrue(data)

        # Check axis organized correctly
        self.assertEqual(data.data.shape, (3,6,3))

        # Check that times are loaded
        self.assertEqual(list(data.data.keys().values), [1,3,7])

        # Check data
        y = data['y']
        self.assertEqual(y[3][4], 8)
        self.assertEqual(y[7][4], 16)

    def testBasicNoTime(self):
        """
        Test that if a file is loaded w/o error (when no time).
        """
        self.copyfiles(copytime=False)

        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertEqual(data.filename, os.path.basename(self._pattern))
        self.assertFalse(data._timedata)
        self.assertTrue(data)

        # Check axis organized correctly
        self.assertEqual(data.data.shape, (3,6,3))

        # Check that times are loaded
        self.assertEqual(list(data.data.keys().values), [0,1,2])

        # Check data
        y = data['y']
        self.assertEqual(y[1][4], 8)
        self.assertEqual(y[2][4], 16)

    def testEmptyUpdateRemove(self):
        """
        Test that non-exist file can be supplied, loaded, and removed.
        """

        # Create object w/o data
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertEqual(data.filename, os.path.basename(self._pattern))
        self.assertFalse(data._timedata)
        self.assertFalse(data)

        # Update data
        self.copyfiles()
        data.update()
        self.assertTrue(data._timedata)
        self.assertTrue(data)

        # Check axis organized correctly
        self.assertEqual(data.data.shape, (3,6,3))
        self.assertEqual(list(data.data.keys().values), [1,3,7])
        y = data['y']
        self.assertEqual(y[3][4], 8)
        self.assertEqual(y[7][4], 16)

        # Remove data
        self.tearDown()
        data.update()
        self.assertFalse(data._timedata)
        self.assertFalse(data)
        self.assertTrue(data['y'].empty)

    def testOldData(self):
        """
        Test that old data is not loaded
        """

        # Load the files
        self.copyfiles()
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertTrue(data)
        self.assertEqual(data.data.shape, (3,6,3))

        # Make the last file old
        time.sleep(2) # wait so new files have newer modified times
        mooseutils.touch('vpp_000.csv')
        mooseutils.touch('vpp_001.csv')

        # Update and make certain data structure is smaller
        data.update()
        self.assertTrue(data)
        self.assertEqual(data.data.shape, (2,6,3))
        self.assertEqual(list(data.data.keys().values), [1,3])

        # Test data
        y = data['y']
        self.assertEqual(y[3][4], 8)

        # Touch 3 so that, it should show up then
        time.sleep(1)
        mooseutils.touch('vpp_002.csv')
        data.update()
        self.assertTrue(data)
        self.assertEqual(data.data.shape, (3,6,3))

    def testRemoveData(self):
        """
        Test that removing a file is handled correctly.
        """

        # Load the files
        self.copyfiles()
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertTrue(data)
        self.assertEqual(data.data.shape, (3,6,3))

        # Remove the middle file
        os.remove('vpp_001.csv')

        # Update and check results
        data.update()
        self.assertTrue(data)
        self.assertEqual(data.data.shape, (2,6,3))
        self.assertEqual(list(data.data.keys().values), [1,7])

        # Test data
        y = data['y']
        self.assertEqual(y[1][4], 4)
        self.assertEqual(y[7][4], 16)

    def testTimeAccess(self):
        """
        Test that time based data access is working.
        """
        self.copyfiles()
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertTrue(data)

        # Test that newest data is returned
        y = data('y')
        self.assertEqual(y[4], 16)

        # Test that older data can be loaded
        y = data('y', time=3)
        self.assertEqual(y[4], 8)

        # Test that bisect returns value even if time is exactly correct
        y = data('y', time=3.3)
        self.assertEqual(y[4], 8)

        # Test that beyond end returns newest
        y = data('y', time=9999)
        self.assertEqual(y[4], 16)

        # Test time less than beginning returns first
        y = data('y', time=0.5)
        self.assertEqual(y[4], 4)

        # Test that disabling bisect returns empty
        y = data('y', time=3.3, exact=True)
        self.assertTrue(y.empty)

    def testVariables(self):
        """
        Check variable names.
        """
        self.copyfiles()
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertTrue(data)
        self.assertIn('x', data.variables())
        self.assertIn('y', data.variables())

    def testRepr(self):
        """
        Test the 'repr' method for writing scripts is working.
        """

        # Load the files
        self.copyfiles()
        data = mooseutils.VectorPostprocessorReader('vpp_*.csv')
        self.assertTrue(data)

        # Get script text
        output, imports = data.repr()

        # Append testing content
        output += ["print 'SHAPE:', data.data.shape"]
        output += ["print 'VALUE:', data['y'][3][4]"]

        # Write the test script
        script = '{}_repr.py'.format(self.__class__.__name__)
        with open(script, 'w') as fid:
            fid.write('\n'.join(imports))
            fid.write('\n'.join(output))

        # Run script
        self.assertTrue(os.path.exists(script))
        out = subprocess.check_output(['python', script])

        # Test for output
        self.assertIn('SHAPE: (3, 6, 3)', out)
        self.assertIn('VALUE: 8', out)

        # Remove the script
        os.remove(script)

if __name__ == '__main__':
    unittest.main(module=__name__, verbosity=2)
