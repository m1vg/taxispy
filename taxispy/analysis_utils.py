from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import numpy as np
import pandas as pd
import time # For eval_fitness_function timing
from .detect_peaks import detect_peaks # Assuming detect_peaks.py is in the same directory

def transform_parameters(parameters, bounds):
    """
    Transforms normalized genetic algorithm parameters to their actual scales.

    Args:
        parameters (list or tuple): A list of normalized parameter values (typically 0-10 range from DEAP).
                                   Expected order: [norm_frames_av, norm_smooth, norm_acc_threshold].
        bounds (list or tuple): A list of upper bounds for each parameter.
                               Expected order: [max_frames_av, max_smooth, max_acc_threshold].

    Returns:
        tuple: (frames_av, smooth, acceleration_threshold)
               frames_av (int): Number of frames for averaging.
               smooth (int): Number of smoothing cycles.
               acceleration_threshold (float): Threshold for acceleration peaks.
    """
    # Ensure parameters and bounds have the expected length if necessary, or rely on caller.
    # Original logic: param_val * bound_val / 10
    # Assuming parameters are scaled 0-10, and bounds are the max real values.
    # If parameters from DEAP are 0-1, the division by 10 might be different.
    # The original InitGui used random.randint(1,10) for individual generation,
    # suggesting parameters are indeed in a range that makes sense with /10.
    
    frames_av = int(abs(parameters[0] * bounds[0] / 10))
    smooth = int(abs(parameters[1] * bounds[1] / 10))
    acceleration_threshold = abs(parameters[2] * bounds[2] / 10)
    return frames_av, smooth, acceleration_threshold

def calculate_vel(t_particle_df, frames_second, pixels_micron):
    """
    Calculates instantaneous velocity and acceleration for a single particle's trajectory.

    Args:
        t_particle_df (pd.DataFrame): DataFrame for a single particle, must contain 'x' and 'y' columns,
                                      and be indexed by frame number.
        frames_second (float): Frames per second for the video.
        pixels_micron (float): Pixels per micron conversion factor.

    Returns:
        tuple: (vel_df, acc_vel_df)
               vel_df (pd.DataFrame): DataFrame of instantaneous linear velocity, indexed by frame.
               acc_vel_df (pd.DataFrame): DataFrame of instantaneous linear acceleration, indexed by frame.
    """
    if t_particle_df.empty or len(t_particle_df) < 2:
        # Return empty DataFrames with expected column name if particle df is too short
        particle_id_col = [t_particle_df.name] if hasattr(t_particle_df, 'name') else [0] # Attempt to get particle ID for column name
        return pd.DataFrame(columns=particle_id_col), pd.DataFrame(columns=particle_id_col)

    # Ensure particle_id_col is a list/iterable for DataFrame columns argument
    particle_id_col_name = [t_particle_df.columns[0]] if hasattr(t_particle_df, 'columns') and len(t_particle_df.columns) > 0 else [0]
    if 'particle' in t_particle_df.columns: # If t_particle_df is from t1 before selecting a particle
        particle_id_val = t_particle_df['particle'].iloc[0]
    else: # If t_particle_df is already for a single particle (e.g. a Series or specific DF structure)
         # Attempt to get particle ID for column name from DataFrame name or use a default
        particle_id_val = t_particle_df.name if hasattr(t_particle_df, 'name') else 0


    x = t_particle_df['x']
    y = t_particle_df['y']
    
    # Use .index directly if t_particle_df is already indexed by frame.
    # If 'frame' is a column, it should be set as index before calling this function.
    frame_index = t_particle_df.index

    vel_data = pd.Series(np.nan, index=frame_index, name=particle_id_val)
    acc_vel_data = pd.Series(np.nan, index=frame_index, name=particle_id_val)

    if pixels_micron == 0: # Avoid division by zero
        return pd.DataFrame(vel_data), pd.DataFrame(acc_vel_data)

    # Calculate displacement (dx, dy)
    dx = x.diff()
    dy = y.diff()

    # Distance between consecutive frames
    dist = np.sqrt(dx**2 + dy**2)
    
    # Velocity: distance / time_per_frame, scaled by pixels_micron
    # time_per_frame is 1.0 / frames_second
    vel_data = (dist * frames_second) / pixels_micron
    
    # Acceleration: change in velocity / time_per_frame
    acc_vel_data = vel_data.diff() * frames_second
    
    vel_df = pd.DataFrame(vel_data)
    acc_vel_df = pd.DataFrame(acc_vel_data)
    
    # First velocity value and first two acceleration values will be NaN due to diff()
    return vel_df.iloc[1:], acc_vel_df.iloc[2:]


def calculate_average_vel(vel_df, threshold=4):
    """
    Calculates a rolling average of velocity.

    Args:
        vel_df (pd.DataFrame): DataFrame of velocities for one or more particles.
        threshold (int): The window size for the rolling average.

    Returns:
        pd.DataFrame: DataFrame of smoothed velocities.
    """
    if threshold <= 0: # Or handle as an error
        return vel_df.copy()
    
    # Using rolling mean, ensuring enough periods for edge cases.
    # The original code had custom logic for the start of the series.
    # df.rolling(window, min_periods=1).mean() handles edges by calculating mean on available data.
    average_vel_df = vel_df.rolling(window=threshold, min_periods=1).mean()
    return average_vel_df


def calculate_av_acc(av_vel_df, frames_second):
    """
    Calculates acceleration from averaged velocity.

    Args:
        av_vel_df (pd.DataFrame): DataFrame of averaged velocities.
        frames_second (float): Frames per second for the video.

    Returns:
        pd.DataFrame: DataFrame of accelerations based on averaged velocity.
    """
    if frames_second == 0: # Avoid division by zero
        return pd.DataFrame(np.nan, index=av_vel_df.index, columns=av_vel_df.columns)

    # Acceleration: change in velocity / time_per_frame
    # time_per_frame is 1.0 / frames_second
    acc_vel_df = av_vel_df.diff() * frames_second
    return acc_vel_df.iloc[1:] # First value will be NaN


def get_peaks_data(t1_particle_df, frames_second, abs_av_acc_series, acceleration_threshold):
    """
    Detects peaks in acceleration data to determine changes of direction (tumbles).

    Args:
        t1_particle_df (pd.DataFrame): DataFrame for a single particle from the main trajectory data (self.t1).
                                      Used to determine total duration of the trajectory.
        frames_second (float): Frames per second.
        abs_av_acc_series (pd.Series): Series of absolute averaged acceleration values for the particle.
        acceleration_threshold (float): Minimum peak height for detecting acceleration peaks.

    Returns:
        tuple: (change_dir, duration, frequency)
               change_dir (float): Number of changes of direction (tumbles).
               duration (float): Total duration of the particle's trajectory in seconds.
               frequency (float): Tumbling frequency (change_dir / duration).
    """
    if abs_av_acc_series.empty or abs_av_acc_series.isna().all():
        return 0.0, 0.0, 0.0

    # Ensure series is numpy array for detect_peaks
    data = abs_av_acc_series.dropna().values 
    peakind = detect_peaks(data, mph=acceleration_threshold)
    number_peaks = len(peakind)

    # Original logic: "this is done in order to consider changes of direction that consist
    # of one stop event followed by swimming with a slow velocity"
    if number_peaks == 1:
        number_peaks = 2

    # Calculate duration from the original particle trajectory data
    if t1_particle_df.empty or 'frame' not in t1_particle_df.columns:
        duration = 0.0
    else:
        min_frame = t1_particle_df['frame'].min()
        max_frame = t1_particle_df['frame'].max()
        if frames_second > 0:
            duration = (max_frame - min_frame) / frames_second
        else:
            duration = 0.0

    change_dir = np.floor(number_peaks / 2)

    if duration > 0:
        frequency = change_dir / duration
    else:
        frequency = 0.0
        # Avoid division by zero if duration is zero, also means change_dir is effectively 0 in terms of rate
        if change_dir > 0 : # If there are changes but no duration, this is an edge case
             frequency = np.inf # Or handle as an error / undefined

    return change_dir, duration, frequency


def core_eval_fitness(t1_df, particle_id, real_chng_dir, frames_second, pixels_micron, 
                      frames_av_param, smooth_param, acc_thresh_param):
    """
    Core logic for evaluating the fitness of a single particle's parameters.
    This function calculates the number of estimated changes of direction based on
    the provided parameters and compares it to the real number of changes.

    Args:
        t1_df (pd.DataFrame): The complete trajectory DataFrame containing data for all particles.
        particle_id (int): The ID of the particle to evaluate.
        real_chng_dir (int): The observed (real) number of changes of direction for this particle.
        frames_second (float): Video frames per second.
        pixels_micron (float): Pixels per micron conversion factor.
        frames_av_param (int): Number of frames for averaging velocity (smoothing window).
        smooth_param (int): Number of smoothing cycles.
        acc_thresh_param (float): Acceleration threshold for peak detection.

    Returns:
        float: The squared error between estimated and real changes of direction.
    """
    particle_data = t1_df[t1_df['particle'] == particle_id]
    if particle_data.empty or len(particle_data) < 2 : # Need at least 2 points for velocity
        return np.inf # Or a large error value if particle data is insufficient

    # Set 'frame' as index if it's not already, for time-series operations
    # Assuming t1_df has 'frame' as a column from trackpy
    if 'frame' in particle_data.columns:
         # Keep original index in a column if needed elsewhere, though not for these calcs
        # particle_data = particle_data.set_index('frame', drop=False) 
        # For calculations, better to not drop, so original frame numbers are preserved if index is reset later
        particle_data_indexed = particle_data.set_index('frame')
    else: # If 'frame' is already the index
        particle_data_indexed = particle_data

    vel_df, _ = calculate_vel(particle_data_indexed, frames_second, pixels_micron)
    if vel_df.empty:
        return np.inf # Cannot proceed if velocity calculation fails

    # Smooth velocity vector
    av_vel_df = vel_df
    if smooth_param > 0 and frames_av_param > 0:
        current_av_vel = vel_df
        for _ in range(smooth_param): # Apply smoothing iteratively
            current_av_vel = calculate_average_vel(current_av_vel, threshold=frames_av_param)
        av_vel_df = current_av_vel
    
    if av_vel_df.empty:
         return np.inf

    # Calculate acceleration from smoothed velocity
    av_acc_df = calculate_av_acc(av_vel_df, frames_second)
    if av_acc_df.empty:
        return np.inf # Cannot proceed if acceleration calculation fails

    # Get number of changes of direction
    # For get_peaks_data, we need the original particle_data (not indexed by frame necessarily,
    # but the one that has the 'frame' column for duration calculation).
    # And the acceleration series should be just the values for the specific particle.
    # av_acc_df is likely a single column DataFrame here.
    
    # Ensure abs_av_acc_series is a Series.
    # vel_df, av_acc_df from above functions return df with particle_id as column name.
    # If particle_id is not in av_acc_df.columns (e.g. if it was a default like 0),
    # we need to handle this robustly.
    # Assuming av_acc_df has one column, get it as a Series:
    if not av_acc_df.columns.empty:
        abs_av_acc_series = abs(av_acc_df[av_acc_df.columns[0]])
    else: # Should not happen if vel_df was not empty
        return np.inf

    est_chng_dir, _, _ = get_peaks_data(particle_data, # Original df for this particle for duration
                                        frames_second, 
                                        abs_av_acc_series, 
                                        acc_thresh_param)
    
    error = (est_chng_dir - real_chng_dir)**2
    return error

def full_eval_fitness_function(t1_df, training_data_df, frames_second, pixels_micron, 
                               bounds, individual_params):
    """
    Evaluates the fitness of an individual (a set of parameters) across a training set of particles.
    The fitness is typically the sum of squared errors between estimated and real changes of direction.

    Args:
        t1_df (pd.DataFrame): The complete trajectory DataFrame.
        training_data_df (pd.DataFrame): DataFrame containing training data. Must have particle IDs
                                         as index and a 'real_chng_dir' column.
        frames_second (float): Video frames per second.
        pixels_micron (float): Pixels per micron conversion factor.
        bounds (list or tuple): Upper bounds for parameters ([max_frames_av, max_smooth, max_acc_threshold]).
        individual_params (list or tuple): Normalized parameters for the individual from the GA.

    Returns:
        tuple: (total_error, processing_time) - Total squared error and time taken for evaluation.
               DEAP expects a tuple for fitness values.
    """
    frames_av, smooth, acceleration_threshold = transform_parameters(individual_params, bounds)
    
    # Ensure frames_av is not zero, as it's used as a window size.
    frames_av = frames_av if frames_av > 0 else 1 

    total_error = 0
    tic = time.time()

    for particle_id, row in training_data_df.iterrows():
        real_chng_dir = row['real_chng_dir']
        error = core_eval_fitness(t1_df, particle_id, real_chng_dir, frames_second, pixels_micron,
                                  frames_av, smooth, acceleration_threshold)
        total_error += error
        
    toc = time.time()
    # DEAP expects a tuple, so if only one objective, it's (value,)
    return total_error, toc - tic # Returning processing time as a potential second objective if needed by DEAP setup
