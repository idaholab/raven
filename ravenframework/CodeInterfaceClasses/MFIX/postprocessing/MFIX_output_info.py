import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Usage: MFIX_output_info.py SIM_PATH POSTPROCESSING_PATH")
    sys.exit(1)

SIM_PATH = sys.argv[1]
POSTPROCESSING_PATH = sys.argv[2]

input_info = pd.read_csv(f'{SIM_PATH}/Input_file_data.csv')

avg_velocity = np.zeros([input_info.shape[0]])

for i in tqdm(range(input_info.shape[0])):  # loop through the run folders 
    try:
        velocities = np.genfromtxt(f'{POSTPROCESSING_PATH}/{i + 1}/Results/Average_Y_Velocity.csv', delimiter=',').astype(float)
        velocities = velocities[velocities[:, 0].argsort()]
        avg_velocity[i] = np.average(velocities[100:np.size(velocities[:, 0]), -1])
    except:
        avg_velocity[i] = 999999
df = pd.DataFrame(input_info)
df.insert(df.shape[1], 'Average Edge Velocity', avg_velocity, True)
df.to_csv(f'{POSTPROCESSING_PATH}/run_results.csv', index=False)
