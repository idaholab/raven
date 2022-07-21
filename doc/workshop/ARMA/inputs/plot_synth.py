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
Simple plotting script to create visualizations of ARMA workshop material
"""
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    try:
        synthFilepath = os.path.abspath(sys.argv[1])
        realFilepath = os.path.abspath(os.path.join(os.path.dirname(synthFilepath), "dataSet_0.csv"))
    except IndexError:
        print("ERROR: Filepath to synthetic data not provided")


    synth = pd.read_csv(synthFilepath)
    real = pd.read_csv(realFilepath)

    fig, ax = plt.subplots()
    ax.plot(synth.iloc[:, 0], synth.iloc[:, 1], color="goldenrod", label='Synthetic Signal')
    ax.plot(real.iloc[:, 0], real.iloc[:, 1], label='Original Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    outputFilepath = os.path.join(os.path.dirname(real_fp), "results.png")
    plt.savefig(outputFilepath)


if __name__ == '__main__':
    main()
