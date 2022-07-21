#!/usr/bin/env python

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    try:
        synth_fp = os.path.abspath(sys.argv[1])
        real_fp = os.path.abspath(os.path.join(os.path.dirname(synth_fp), "dataSet_0.csv"))
    except IndexError:
        print("ERROR: Filepath to synthetic data not provided")


    synth = pd.read_csv(synth_fp)
    real = pd.read_csv(real_fp)

    fig, ax = plt.subplots()

    ax.plot(synth.iloc[:, 0], synth.iloc[:, 1], color="goldenrod", label='Synthetic Signal')
    ax.plot(real.iloc[:, 0], real.iloc[:, 1], label='Original Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    output_fp = os.path.join(os.path.dirname(real_fp), "results.png")
    plt.savefig(output_fp)


if __name__ == '__main__':
    main()
