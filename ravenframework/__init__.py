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
Created on September 16, 2015
@author: maljdp
"""

import os
import sys

# On Linux, pre-load the env's libstdc++ with RTLD_GLOBAL before any native library imports.
# Why: when Python launches on hosts whose system /lib64/libstdc++.so.6 is older than the
# conda env's GCC-built shared objects (e.g. libicui18n.so.78 needs CXXABI_1.3.15 from
# GCC 13/14), the dynamic linker binds libstdc++ from /lib64 first and later imports that
# touch ICU (e.g. _sqlite3 -> libicui18n) fail with "version 'CXXABI_1.3.15' not found".
# Pre-loading the env's libstdc++ as RTLD_GLOBAL guarantees its symbols are in the global
# scope before any C extension shows up. Covers unit tests that import ravenframework
# directly without going through raven_framework.py.
if sys.platform.startswith('linux'):
    import ctypes
    _env_libstdcxx = os.path.join(sys.prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(_env_libstdcxx):
        try:
            ctypes.CDLL(_env_libstdcxx, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

if sys.version_info[0] > 2:
    from .CustomDrivers.PythonRaven import Raven # allows from ravenframework import Raven


# This file is necessary so that the sub-modules understand the correct hierarchy
# of things. Once everything is in sub-modules we can possibly do some things
# with RAVEN in its entirety as a module, but for now this file can remain
# empty.
