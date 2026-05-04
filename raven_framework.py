#!/usr/bin/env python
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
Created on Feb 14, 2022

@author: cogljj

This is a package that properly imports Driver and runs it.
"""
import os
import sys

# Pre-import tensorflow before anything else if it is available.
# Reason: tf-keras 2.21 + tensorflow 2.21 ship Abseil v20250814 statically vendored, and
# PyArrow's libarrow.<version>.dylib also ships Abseil with the same versioned symbol names.
# Whichever library is loaded first wins the dynamic-linker symbol resolution race; if PyArrow
# loads first (which happens transitively through pandas/xarray/dask in RAVEN), TensorFlow
# ends up calling PyArrow's Abseil with a different struct layout for Mutex/Notification, and
# the eager kernel deadlocks forever in AbslInternalPerThreadSemWait. Importing TF first
# makes its Abseil symbols win and PyArrow uses TF's copy successfully (verified: 0.11s vs
# >4-min hang). Cost is a one-time ~1.5s startup penalty on TF-less tests; acceptable trade.
#
# TF_USE_LEGACY_KERAS must be set before tf imports so tf.keras routes to the tf_keras (Keras 2)
# legacy package; the bundled Keras 3 dropped tf.keras.optimizers.legacy plus several layers
# (LocallyConnected1D/2D, etc.) that RAVEN's KerasBase still references.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
try:
    import tensorflow  # noqa: F401
except ImportError:
    pass

from ravenframework.Driver import main
if __name__ == '__main__':
  sys.exit(main(True))
