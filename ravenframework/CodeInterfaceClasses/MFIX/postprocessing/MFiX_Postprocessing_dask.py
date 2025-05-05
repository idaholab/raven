"""
---------------------------------------------------------------------------------

Author      : Denver Conger, Ben Stoyer
Date        : 7/23/2024

Description:
    This script processes directories of mfix produced data to compute average
    gas volume fractions, errors relative to a reference curve, and bed height data.
    It also generates and saves plots for visualizing the computed values.

    The script uses Dask for parallel processing and several other libraries including NumPy, Matplotlib,
    and PyVista for data handling and or visualization.

Usage:
    1. Make sure the Dask cluster is running and the scheduler information is saved
       in 'dask_scheduler_info.txt'.

    2. Run the script from the command line with the following arguments:

        - To process directories:
            python script_name.py --dir <directory_path>

        - To fix directories listed in 'missing_dirs.txt':
            python script_name.py --dir <directory_path> --fix-missing

    3. If 'missing_dirs.txt' is used, make sure it contains the relative paths
       to be processed, one per line.

    4. Output is saved in the 'congds_postprocessing' directory inside the provided
       directory path. This includes CSV files with particle information and plots.

---------------------------------------------------------------------------------
"""

"""
Run*/                                              
  1/                       ┌─────────────────────┐ 
  2/                       │                     │ 
  3/  ──────────────────►  │  Process File Func  │ 
    1.vtp────────┐         │                     │ 
    2.vtp        │         │ Runs per simulation │ 
    ...          │         │                     │ 
    250.vtp      │         └──────────┬──────────┘ 
                 │                    │            
    1.vtu────┐   │         ┌──────────────────────┐
    2.vtu    │   │         │                      │
    ...      │   │         │  Process Single File │
    250.vtu  │   │         │  Func                │
  4/         └───►────────►│(Runs 250 times for   │
  ...                      │ the vtu and vtp pairs│
  1600/                    │ for each simulation) │
                           └──────────────────────┘
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import glob
import os
from tqdm import tqdm
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed, get_client
#from dask.distributed import retry
import dask.array as da
from numba import jit, njit
import time
import traceback
import argparse
from loguru import logger
import random
import tempfile
import shutil
import numpy as np
import gc

# Constants
delta_h = 0.001
delta_vol = 0.1
slope_stop = delta_vol / delta_h
d_part = 2.5e-4
r_tol = d_part / 1000
r_hem = 0.0045
h_off = r_hem / np.tan(np.pi / 6)
n_y_mesh = 115

def numpy_moving_avg(x, w):
    """
    Computes the moving average of an array using NumPy.

    Args:
        x (np.ndarray): Input array of data.
        w (int): The window size for the moving average.

    Returns:
        np.ndarray: The array containing the moving averages.
    """
    # Use convolution with a window of ones divided by the window size to compute the moving average
    
    return np.convolve(x, np.ones(w), 'valid') / w


def process_volume_fraction(cent_coord, vol_frac, bins, n_y_mesh):
    """
    Computes the average volume fraction for each calculated bin. 

    Args:
        cent_coord (np.ndarray): Array of cell center coordinates.
        vol_frac (np.ndarray): Array of volume fractions.
        bins (np.ndarray): Array of bin edges for the mesh.
        n_y_mesh (int): Number of bins along the y-axis.

    Returns:
        np.ndarray: Array containing the average volume fractions for each bin.
    """
    # Initialize the output array for storing average volume fractions and related data
    avg_volfrac = np.zeros((n_y_mesh, 4))
    avg_volfrac[:, 0] = bins
    digitized = np.digitize(cent_coord[:, 1], bins)

    for i in range(1, n_y_mesh):
        mask = digitized == i
        avg_volfrac[i, 1] = np.sum(vol_frac[mask])
        avg_volfrac[i, 2] = mask.sum()  # More efficient sum

    # Efficiently calculate average volume fractions
    non_zero_mask = avg_volfrac[:, 2] != 0
    avg_volfrac[non_zero_mask, 3] = avg_volfrac[non_zero_mask, 1] / avg_volfrac[non_zero_mask, 2]

    return avg_volfrac

@dask.delayed
def calculate_edge_velocity_profile(bed_edge_dict, h_bed_avg):
    """
    Calculate the edge velocity profile for bed particles.

    This function processes the edge particles of the bed, binning them by height,
    and calculating the average velocity and slope of the velocity profile across
    the bed height. It also tracks the edge particle velocities over time.

    Args:
        bed_edge (np.ndarray): The array of edge particle data with positions and velocities.
        heights (np.ndarray): The array containing the processed bed heights at each time step.
        time_var (float): The current time variable for associating with the velocity data.

    Returns:
        list: A list of average velocities at the current time step.
        list: A list of average velocity slopes at the current time step.
        np.ndarray: An array of the average y-velocity for all edge particles at different heights.
    """
    all_avg_vel_list, all_avg_slope_list, all_avg_vel_alltime_list, all_edge_space_sorted_2_columns_list = [],[],[],[]
    avg_vel = np.zeros([1, 2])  # create blank average velocity matirx
    avg_slope = np.zeros([1, 2])  # create blank average slop matrix
    avg_vel_alltime = np.zeros([0, 3])
    for time_var, bed_edge in bed_edge_dict.items():
        # Calculate the average bed height from the heights array
        # n_bed = 50  # Number of bins for height
    
        h_percent_high = 0.8  # Upper percentage for height cut-off
        height_bin = 0.01
        max_height = h_bed_avg
        n_bed = int(np.ceil(max_height / height_bin))
        
        height_bins = np.linspace(0, max_height, num=n_bed, endpoint=True)

        valid_mask = (bed_edge[:, 1] >= 0) & (bed_edge[:, 1] < max_height)
        valid_particles = bed_edge[valid_mask]
        bin_indices = np.searchsorted(height_bins, valid_particles[:, 1], side='left')
        bin_indices = np.clip(bin_indices, 0, n_bed - 1)
        
        binned_velocities = np.zeros((n_bed, 3))
        np.add.at(binned_velocities, bin_indices, valid_particles[:, 3:6])
        bin_counts = np.bincount(bin_indices, minlength=n_bed)
        
        nonzero_bins = bin_counts > 0
        average_velocities = np.zeros_like(binned_velocities)
        average_velocities[nonzero_bins] = binned_velocities[nonzero_bins] / bin_counts[nonzero_bins][:, None]
        
        valid_height_bins = height_bins[nonzero_bins]
        
        # try:
        #     slopes_y = np.gradient(average_velocities[nonzero_bins, 1], valid_height_bins)

        avg_vel_alltime_array = np.column_stack((np.full(np.sum(nonzero_bins), time_var), average_velocities[nonzero_bins, 1], valid_height_bins))
        avg_vel = np.append(avg_vel, [[time_var, np.mean(average_velocities[nonzero_bins, 1])]], axis=0)

        shape = valid_height_bins.shape

        # Create an array filled with time_var, matching the shape of the other arrays
        time_var_array = np.full(shape, time_var)
        edge_space_sorted_2_columns = np.stack((valid_height_bins, average_velocities[nonzero_bins, 1], time_var_array), axis=1)
        all_avg_vel_alltime_list.append(avg_vel_alltime_array)
        all_edge_space_sorted_2_columns_list.append(edge_space_sorted_2_columns)
    
    if all_edge_space_sorted_2_columns_list:  # Check that the list is not empty
        all_edge_space_sorted_2_columns_array = np.concatenate(all_edge_space_sorted_2_columns_list, axis=0)
    else:
        print("No edge space data to concatenate.")
        # You might want to handle this case appropriately, e.g., by creating an empty array
        all_edge_space_sorted_2_columns_array = np.empty((0, 3))
        
    return all_edge_space_sorted_2_columns_array


def save_plots(avg_volfrac, cow, h_bed, err, EP_G_bed, subdir, save_dir):
    """
    Generates and saves plots for average volume fractions and errors.

    Args:
        avg_volfrac (np.ndarray): Array of average volume fractions.
        cow (np.ndarray): Moo-ving average array.
        h_bed (float): Detected bed height.
        err (np.ndarray): Array of errors.
        EP_G_bed (float): Detected bed gas volume fraction.
        name (str): Name of the file to save plots.
        subdir (str): Subdirectory path for saving plots.

    Returns:
        None
    """
    # Define the output directory for saving plots
    output_dir = f'/projects/MFIX/{save_dir}/{os.path.basename(subdir)}/Results'
    os.makedirs(output_dir, exist_ok=True)  # make sure the directory exists

    # Plot and save the average gas volume fraction vs height
    plt.close('all')
    plt.plot(np.asarray(avg_volfrac[:, 0]), np.asarray(avg_volfrac[:, 3]), 'b-')
    plt.plot(np.asarray(avg_volfrac[:len(cow), 0]), np.asarray(cow), 'g-o')
    plt.plot(np.asarray(h_bed), np.asarray(EP_G_bed), 'ro')
    plt.xlabel('Height')
    plt.ylabel('Average Gas Volume Fraction')
    #plt.savefig(f'{output_dir}/{name}_EP_G_figure.png')
    plt.close()

    # Plot and save the error vs height
    plt.plot(np.asarray(avg_volfrac[:len(cow), 0]), np.asarray(err), 'b-o')
    #plt.savefig(f'{output_dir}/{name}_EP_G_figure_err.png')
    plt.close()
    return True

def pv_read_delayed(file_path):
    """
    THe whole reason this is a function is because dask 
    does some weird stuff under the hood with how it calculated 
    what it does and when and I found that by putting the read 
    logic within its own function the dask delayed logic was able 
    to compute stuff slightly faster.

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pv.read(file_path)


def process_error(avg_volfrac, cow):
    """
    Computes the error from a given tolerance between the average volume 
    fraction and the moo-ving average version. Although I do want to talk to stoyer about the 
    usefulness of this function,
    this seems to be really old logic and it may be integral to the calculations but I cant be certain.

    Args:
        avg_volfrac (np.ndarray): Array of average volume fractions.
        cow (np.ndarray): Moo-ving average array.
        n_y_mesh (int): Number of bins along the y-axis.
        bins (np.ndarray): Array of bin edges for the mesh.

    Returns:
        tuple: A tuple containing arrays of errors and detected bed height information.
    """
    # Compute the absolute error between average volume fractions and the reference curve
    err = np.abs(avg_volfrac[:len(cow), 3] - cow)
    err_tol = 0.0001
    EP_G_bed = h_bed = None

    # Detect the bed height based on error tolerance
    for i in range(1, len(err)):
        if np.mean(err[i:i + 10]) < err_tol and err[i - 1] > err_tol:
            EP_G_bed = cow[i - 1]
            h_bed = avg_volfrac[i - 1, 0]
            break

    # make sure the bed height does not exceed the maximum height in the bins
    h_bed = min(h_bed, np.max(avg_volfrac[:, 0])) if h_bed is not None else np.max(avg_volfrac[:, 0])

    return err, EP_G_bed, h_bed


@dask.delayed
def process_single_file(j, subdir, save_dir, d_part, r_tol, r_hem, h_off, n_y_mesh):
    """
    Processes a single simulation data file.

    Args:
        j (str): The file path to the vtp file.
        subdir (str): The subdirectory containing the files.
        delta_h (float): Height increment for processing.
        delta_vol (float): Volume increment for processing.
        slope_stop (float): Slope stop criterion.
        d_part (float): Particle diameter.
        r_tol (float): Radial tolerance.
        r_hem (float): Hemispherical radius.
        h_off (float): Height offset.
        n_y_mesh (int): Number of bins along the y-axis.

    Returns:
        np.ndarray: Array containing the processed heights.
    """
    try:
        os.chdir(subdir)
        output_dir = f'/projects/MFIX/{save_dir}/{os.path.basename(subdir)}/Results'
        os.makedirs(output_dir, exist_ok=True)  # make sure the directory exists
        heights = np.zeros((1, 2))
        file_id_2 = j.split('C_')[-1].split('.')[0]
        name = os.path.splitext(os.path.basename(j))[0]
        file_cell = f"X_SLICE_{file_id_2}.vtu"

        # Check if the original files exist
        if not os.path.exists(file_cell) or not os.path.exists(j):
            raise FileNotFoundError(f"File {file_cell} or {j} not found.")

        # Read the original files directly
        mesh = pv_read_delayed(file_cell)
        part_base = pv_read_delayed(j)

        # Process the data from the files
        part = np.array(part_base.points)
        part_vel = np.array(part_base.get_array("Velocity"))
        cent_coord = np.array(mesh.cell_centers().points)
        vol_frac = np.array(mesh.get_array('EP_G'))

        y_min_cell = np.min(cent_coord[:, 1])
        y_max_cell = np.max(cent_coord[:, 1])
        bins = np.linspace(y_min_cell, y_max_cell, num=n_y_mesh)

        avg_volfrac = process_volume_fraction(cent_coord, vol_frac, bins, n_y_mesh)
        cow = numpy_moving_avg(avg_volfrac[:, 3], 5)

        process_error_result = process_error(avg_volfrac, cow)
        err, EP_G_bed, h_bed = process_error_result


        #plot_saved = save_plots(avg_volfrac, cow, h_bed, err, EP_G_bed, subdir, save_dir)

        if len(vol_frac) == len(cent_coord):
            bed_part_mask = part[:, 1] <= h_bed
            # spout_part_mask = ~bed_part_mask
            bed_part = np.hstack((part[bed_part_mask], part_vel[bed_part_mask]))
            # spout_part = np.hstack((part[spout_part_mask], part_vel[spout_part_mask]))
            if len(bed_part) > 0:
                x_bar, y_bar, z_bar = np.mean(bed_part[:, :3], axis=0)
                bed_part[:, :3] -= [x_bar, y_bar, z_bar]
                h_bed -= y_bar
                # spout_part[:, :3] -= [x_bar, y_bar, z_bar]
                y_min = np.min(bed_part[:, 1])
                bed_part[:, 1] -= y_min
                # spout_part[:, 1] -= y_min
                h_bed -= y_min
                mask_edge = (bed_part[:, 0]**2 + bed_part[:, 2]**2)**0.5 >= np.tan(np.pi / 6) * (bed_part[:, 1] + h_off) - d_part - r_tol
                bed_edge = bed_part[mask_edge]
                heights[0, 0] = int(file_id_2) / 100
                heights[0, 1] = h_bed
#                 output_dir = f"/projects/MFIX/{save_dir}/{os.path.basename(subdir)}"
#                 os.makedirs(output_dir, exist_ok=True)
#                 np.savetxt(f"{output_dir}/{name}_EdgeParticlePoints.csv", bed_edge, delimiter=',')
#                 np.savetxt(f"{output_dir}/{name}_BedParticlePoints.csv", bed_part, delimiter=',')
#                 np.savetxt(f"{output_dir}/{name}_SpoutParticlePoints.csv", spout_part, delimiter=',')
        if len(bed_edge) > 1:
            if len(bed_part) > 0:

        
                # return heights, bed_edge, bed_part, spout_part, int(file_id_2) / 100
                return heights, bed_edge, int(file_id_2) / 100
        else:
            return None
    except FileNotFoundError as e:
        logger.error(f"File not found during processing file {j}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {j}: {e}")
        traceback.print_exc()
        return None

def process_file(file_part, subdir, save_dir, max_concurrent_tasks=10):
    """
    Processes multiple files in a subdirectory, with retries for failed tasks.

    Args:
        file_part (list): List of file paths to be processed.
        subdir (str): Subdirectory containing the files.
        retries (int): Number of retry attempts for failed tasks.

    Returns:
        None
    """
    tasks = [process_single_file(j, subdir, save_dir, d_part, r_tol, r_hem, h_off, n_y_mesh) for j in file_part]
    heights_results, edge_space_sorted_2_columns = [], []
    bed_edge_dict = {}

    with tqdm(total=len(tasks), desc=f"Processing {os.path.basename(subdir)}") as progress_bar:
        client = get_client()  # Get the current client
        futures = client.compute(tasks)  # Compute tasks to get actual futures
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    h, be, time_var = result
                    if h.size > 0:
                        heights_results.append(h)
                    bed_edge_dict[time_var] = be
                progress_bar.update(1)
            except Exception as exc:
                logger.error(f"Generated an exception: {exc}")
                progress_bar.update(1)

    if not heights_results:
        raise ValueError("No valid results obtained after computation.")
    all_heights = np.vstack(heights_results)
    h_bed_avg = np.average(all_heights[:, 1])
    tasks_h = [calculate_edge_velocity_profile(bed_edge_dict, h_bed_avg)]

    future_results = client.compute(tasks_h)
    finished_heights_results = client.gather(future_results)
    for result in finished_heights_results:
        if result is not None:
            edge_space_2_columns = result
            edge_space_sorted_2_columns.append(edge_space_2_columns)

    edge_space_sorted_2_columns_np = np.array(edge_space_sorted_2_columns)
    if len(edge_space_sorted_2_columns_np) > 0:
        all_edge_space_sorted_2_columns = np.concatenate(edge_space_sorted_2_columns_np, axis=0)
        logger.debug(f"SHAPE ALL: : {all_edge_space_sorted_2_columns.shape}")
    else:
        all_edge_space_sorted_2_columns = np.empty((0, 3))

    output_dir = f"/projects/MFIX/{save_dir}/{os.path.basename(subdir)}"
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "Avg_vel_per_bin_and_time.csv")
    np.savetxt(results_path, all_edge_space_sorted_2_columns, delimiter=',')

    del tasks, tasks_h, heights_results, bed_edge_dict, finished_heights_results, edge_space_sorted_2_columns_np, all_edge_space_sorted_2_columns, edge_space_2_columns
    gc.collect()



# Main Execution
if __name__ == "__main__":
    
    logger.remove()
    logger.add(sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=True)
    parser.add_argument('--file_dir', '-d', type=str, required=True, help="directory path to process")
    parser.add_argument('--save_dir', '-s', type=str, default='congds_postprocessing',help="Where to save postprocessing data to.")
    parser.add_argument('--batch', '-b', type=int, required=True, help="Batch Size")
    args = parser.parse_args()

    client = Client(scheduler_file="/projects/MFIX/scheduler.json")

    with open('successful_file_nums.txt', 'r') as file:
        subdirs = [os.path.join(args.file_dir, line.strip()) for line in file if line.strip()]

    start_time = time.time()
    max_concurrent_files = 10  # Limit the number of concurrent process_file tasks
    futures = []

    with tqdm(total=len(subdirs), desc="Submitting Tasks") as progress_bar:
        for subdir in subdirs:
            file_part = [os.path.abspath(f) for f in glob.glob(os.path.join(subdir, "BACKGROUND_IC_*.vtp"))]
            if not file_part:
                logger.warning(f"No files found in {subdir}, skipping.")
                continue

            # Scatter the file_part list to avoid large graph warnings
            scattered_file_part = client.scatter(file_part)
            future = client.submit(process_file, scattered_file_part, subdir, args.save_dir, max_concurrent_tasks=40)
            futures.append(future)
            progress_bar.update(1)

            while len(futures) >= max_concurrent_files:
                completed_future = as_completed(futures).__next__()
                try:
                    completed_future.result()
                except Exception as e:
                    logger.error(f"Task failed for {subdir}: {e}")
                futures.remove(completed_future)

    client.gather(futures)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time} seconds")
    print(f"Average execution time per folder: {elapsed_time / len(subdirs)} seconds")