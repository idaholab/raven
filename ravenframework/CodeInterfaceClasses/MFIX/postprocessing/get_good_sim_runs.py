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
  Identification of simulations which either terminated with error message or not properly terminated. 
  Four different cases of Errors are identified. 
"""

# External Modules
import os
import csv
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Lock
import pandas as pd
import numpy as np
# Path to the directory containing simulation folders

def process_file(file_num, dir_path):
    out_file_path = os.path.join(dir_path, str(file_num), "RoundBottomRetort_BLANK.stdout")
    if os.path.isfile(out_file_path):
        with open(out_file_path, "r") as file:
            lines = file.readlines()
            simulation_time, cpu_used, cpu_io_used, wall_time_used = "", "", "", ""
            terminated_properly = True

            # Check the last 10 lines for termination messages
            if len(lines) > 10:
                for line in lines[-10:]:
                    line = line.strip()
                    if "Program Terminated" in line:
                        terminated_properly = False
                    if line.startswith("Final simulation time ="):
                        simulation_time = line.split("=")[1].strip()
                    elif line.startswith("Total CPU used ="):
                        cpu_used = line.split("=")[1].strip()
                    elif line.startswith("Total CPU IO used ="):
                        cpu_io_used = line.split("=")[1].strip()
                    elif line.startswith("Total wall time used ="):
                        wall_time_used = line.split("=")[1].strip()

            # Return data if terminated properly and all required fields are present
            if terminated_properly and simulation_time and cpu_used and cpu_io_used and wall_time_used:
                return (file_num, [file_num, simulation_time, cpu_used, cpu_io_used, wall_time_used])
    else:
        print(f"File {out_file_path} does not exist")
    return None

# Function to process each file for the second loop
def process_mfix_file(file_num, dir_path, vars, result_list):
    mfix_file_path = os.path.join(dir_path, str(file_num), "Test_physics_1a.mfx")
    if os.path.isfile(mfix_file_path):
        row_data = {'subdirectory': file_num}
        with open(mfix_file_path, "r") as file:
            contents = file.readlines()
            for line in contents:
                line = line.strip()
                for var in vars:
                    if var in line:
                        variable, value = line.split('=')
                        row_data[var] = float(value.strip())
        result_list.append(row_data)


def main(dir_path, save_dir, nprocs):
    # List of folder numbers to process
    file_nums = list(range(1, len(next(os.walk(dir_path))[1]) + 1))
    
    # Initialize Manager lists to store processed data and successful file numbers
    manager = Manager()
    out_file_data = manager.list()
    successful_file_nums = manager.list()
    result_list = manager.list()
    lock = Lock()

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, file_num, dir_path): file_num for file_num in file_nums}
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(file_nums), desc="Processing", unit="item"):
            result = future.result()
            if result:
                file_num, data = result
                with lock:
                    out_file_data.append(data)
                    successful_file_nums.append(file_num)

    # Convert managed lists to regular lists for sorting
    out_file_data = list(out_file_data)
    successful_file_nums = list(successful_file_nums)

    # Sort out_file_data by file_num
    out_file_data.sort(key=lambda x: x[0])

    # Write the out file data to a CSV file
    with open(f"{save_dir}/out_file_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Folder", "Final Simulation Time", "Total CPU Used", "Total CPU IO Used", "Total Wall Time Used"])
        writer.writerows(out_file_data)
    
    # Write the successful file numbers to a text file
    with open(f"{save_dir}/successful_file_nums.txt", "w") as txtfile:
        for file_num in sorted(successful_file_nums):
            txtfile.write(f"{file_num}\n")
    
    print("Out file data written to 'out_file_data.csv'")
    print("Successful file numbers written to 'successful_file_nums.txt'")


    with open(f'{save_dir}/successful_file_nums.txt', 'r') as file:
        file_nums = [int(line.strip()) for line in file if line.strip()]
    
    # Initialize an empty DataFrame
    df = pd.DataFrame(index=file_nums)
    
    # Define the variables
    vars = ['mu_g0', 'ro_g0', 'd_p0(1)', 'ro_s0(1)', 'ic_t_g(1)', 'ic_des_sm(2,1)', 'bc_massflow_g(2)']
    
    # Add the 'subdirectory' column with the read file numbers
    df['subdirectory'] = df.index
    
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(process_mfix_file, file_num, dir_path, vars, result_list) for file_num in file_nums]
        
        for future in tqdm(as_completed(futures), total=len(file_nums), desc="Processing MFIX files", unit="file"):
            future.result()  # Ensure any exceptions are raised

    # Update the DataFrame with the results
    for row_data in result_list:
        df.loc[row_data['subdirectory'], vars] = pd.Series(row_data).drop('subdirectory')

    # Reset the index to make 'subdirectory' a column again
    df.reset_index(drop=True, inplace=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(f"{save_dir}/Input_file_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=True)
    parser.add_argument('--dirpath', '-d', type=str, required=True, help="directory path to process")
    parser.add_argument('--save_dir', '-d', type=str, required=True, help="directory path to process")
    parser.add_argument('--nprocs', '-np', type=int, default=30,help="Number of CPU cores to use")
    args = parser.parse_args()
    main(args.dirpath, args.save_dir,args.nprocs)