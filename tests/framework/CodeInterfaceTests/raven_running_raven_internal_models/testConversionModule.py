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



def manipulateScalarSampledVariables(sampledVars):
  """
    This method is aimed to manipulate scalar variables.
    The user can create new variables based on the
    variables sampled by RAVEN
    @ In, sampledVars, dict, dictionary of
          sampled variables ({"var1":value1,"var2":value2})
    @ Out, None, the new variables should be
           added in the "sampledVariables" dictionary
  """
  sampledVars['Models|ROM@subType:SciKitLearn@name:ROM1|coef0']  = sampledVars['Models|ROM@subType:SciKitLearn@name:ROM1|C']/10.0




