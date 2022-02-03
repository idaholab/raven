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
 Created on March 30, 2020
 @author: alfoa
"""
import builtins
import warnings

# line_profiler decorator, @Decorators.timingProfile
## if using kernprof, use "profile" builtin; otherwise, passthrough.
try:
  builtins.profile
  timingProfile = builtins.profile
except (AttributeError, ImportError):
  warnings.warn('Unable to load "timingProfile" decorator; replacing with passthrough ...', ImportWarning)
  timingProfile = lambda f: f

# memory_profiler decorator, @Decorators.memoryProfile
try:
  from memory_profiler import profile as memoryProfile
except (AttributeError, ImportError):
  warnings.warn('Unable to load "memoryProfile" decorator; replacing with passthrough ...', ImportWarning)
  memoryProfile = lambda f: f
