import sys
import getopt
from deap import base, creator, tools, algorithms
from scoop import futures
import random
# from deap import base
# from deap import creator
# from deap import tools
# import time # Not strictly needed here if full_eval_fitness_function returns time
# import numpy as np # Already imported in analysis_utils if needed there
import pandas as pd
# from detect_peaks import detect_peaks # Now imported by analysis_utils

# Import the centralized analysis functions
from . import analysis_utils

# Load data from ExcelFile
# It's good practice to wrap file I/O in try-except
try:
    t1_ = pd.read_excel('deap_excel_data.xlsx', sheet_name='t1_trajectories', index_col=0)
    training_data_ = pd.read_excel('deap_excel_data.xlsx', sheet_name='training_data_details', index_col=0)
    config_df = pd.read_excel('deap_excel_data.xlsx', sheet_name='config_and_bounds', index_col=0) # Assuming Parameter is index

    # Extract config values more robustly
    config_values = config_df['Value'].to_dict() # Convert Series to dict for easier access
    frames_second_ = config_values.get('frames_second')
    pixels_micron_ = config_values.get('pixels_micron')
    obj1_weight_ = config_values.get('obj1_weight', 1.0) # Default if not found
    obj2_weight_ = config_values.get('obj2_weight', 0.0) # Default if not found

    bounds_ = [
        config_values.get('Frames_bound'),
        config_values.get('Smooth_bound'),
        config_values.get('Acceleration_bound')
    ]
    if None in [frames_second_, pixels_micron_] or None in bounds_:
        raise ValueError("One or more required configuration values are missing from deap_excel_data.xlsx.")

except FileNotFoundError:
    print("Error: deap_excel_data.xlsx not found. This file is required for run_deap_scoop.py.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading or parsing deap_excel_data.xlsx: {e}")
    sys.exit(1)


# DEAP Individual and Fitness definition
IND_SIZE = 3 
# Ensure FitnessMin and Individual are created only once if script is re-run in some contexts
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-obj1_weight_, -obj2_weight_))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 10) # GA individuals generated with integers 1-10
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function from analysis_utils
# It directly calls full_eval_fitness_function which contains the core logic
toolbox.register("evaluate", analysis_utils.full_eval_fitness_function, 
                 t1_df=t1_,  # Note: changed from t1 to t1_df for clarity in analysis_utils
                 training_data_df=training_data_, 
                 frames_second=frames_second_, 
                 pixels_micron=pixels_micron_, 
                 bounds=bounds_)
                 # individual_params is passed by DEAP automatically

# Register DEAP operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=5.0, sigma=2.0, indpb=0.2) # Adjusted mu, sigma for 1-10 range
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=50)  # # tools.selNSGA2 # tools.selTournament, tournsize=50


def main(argv):

    try:
        opts, args = getopt.getopt(argv, ["generations=", "population="])
        # print("opts is:", opts)
        # print("args is:", args)

    except getopt.GetoptError:
        print ('testInput.py generations=x populations=y')

        sys.exit(2)

    arg_dic = {}
    for a in args:
        s = a.split('=')
        arg_dic.update({s[0]: int(s[1])})

    generations = arg_dic['generations']
    population = arg_dic['population']


    pop = toolbox.population(n=population)
    halloffame = tools.HallOfFame(2)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True, halloffame=halloffame)

    # bounds_ (loaded from Excel) should be used for transforming parameters
    # halloffame[0] contains the raw parameters (e.g., from 1-10 range) for the best individual.
    frames_av, smooth, acceleration_threshold = analysis_utils.transform_parameters(halloffame[0], bounds_)

    #export results to excel file
    columns = ['Value']
    index = ['frames_av', 'smooth', 'acceleration_threshold']
    values = [frames_av, smooth, acceleration_threshold]
    results_deap = pd.DataFrame(values, index=index, columns=columns)
    results_deap.index.name = 'Parameter'

    # Save raw parameters from halloffame (e.g., values between 1-10)
    # These are the direct output from the GA before transformation.
    raw_halloffame_params = halloffame[0] # Best individual's raw parameters
    index_raw_params = [str(i) for i in range(len(raw_halloffame_params))] # "0", "1", "2"
    
    results_halloffame_raw = pd.DataFrame({'Value': raw_halloffame_params}, index=index_raw_params)
    results_halloffame_raw.index.name = 'RawParameterIndex'


    try:
        with pd.ExcelWriter('results_deap.xlsx') as writer:
            results_deap.to_excel(writer, sheet_name='results_deap') # Transformed params
            results_halloffame_raw.to_excel(writer, sheet_name='halloffame_raw') # Raw params
    except Exception as e:
        print(f"Error writing results_deap.xlsx: {e}")

if __name__ == "__main__":
    main(sys.argv[1:])
