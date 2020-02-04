import sys
import getopt
from deap import base, creator, tools, algorithms
from scoop import futures
import random
# from deap import base
# from deap import creator
# from deap import tools
import time
import numpy as np
import pandas as pd
from detect_peaks import detect_peaks


def transform_parameters(parameters, bounds):

    # unpack and transform
    frames_av = int(abs(parameters[0] * bounds[0]/10))
    smooth = int(abs(parameters[1] * bounds[1]/10))
    acceleration_threshold = abs(parameters[2] * bounds[2]/10)
    return frames_av, smooth, acceleration_threshold


def get_peaks_data(t1, frames_second, vel, mph=10):
    # this function should return the number of change of directions, event duration & frequency of change.
    # get indices
    col = vel.columns
    values = vel[col].dropna().values
    data = np.asanyarray([i[0] for i in values])
    peakind = detect_peaks(data, mph=mph)
    number_peaks = len(peakind)

    # this is done in order to consider changes of direction that consist
    # of one stop event followed by swimming with a slow velocity
    if number_peaks == 1:
        number_peaks = 2

    # get frames for the particle contained in vel.
    particle = vel.columns.values
    t_i = t1[t1['particle'] == particle[0]]

    min_value = t_i['frame.1'].iloc[0]
    max_value = t_i['frame.1'].iloc[-1]
    time = (max_value - min_value) / frames_second

    # frequency of change
    change_dir = number_peaks / 2
    change_dir = np.floor(change_dir)

    frequency = change_dir / time

    return change_dir, time, frequency


def calculate_average_vel(vel, threshold=4):

    average_vel = pd.DataFrame(np.nan, index=vel.index, columns=vel.columns)
    ff = vel.index.values[0]  # first frame
    for frame in vel.index.values:
        if frame < ff + threshold - 1:
            average_vel.loc[frame] = np.mean(vel.loc[:frame].dropna().values)
        else:
            average_vel.loc[frame] = np.mean(vel.loc[frame - threshold + 1:frame].dropna().values)
    return average_vel


def calculate_av_acc(frames_second, vel):

    acc_vel = pd.DataFrame(np.nan, index=vel.index.values[1:], columns=vel.columns)
    for frame in vel.index.values[1:]:
        acc_vel.loc[frame] = (vel.loc[frame] - vel.loc[frame - 1]) * frames_second
    return acc_vel

def calculate_vel(data, particleID, frames_second, pixels_micron):
        # this function gets a data frame containing information of all particles,
        # a desired particle and return velocity and accelereration data frames
        particleID = particleID if (isinstance(particleID, int) or isinstance(particleID, list)) else int(particleID)

        # get t_i for the desired particle
        t_i = data[data['particle'] == particleID]

        # get x and y vectors for the desired particle
        x = t_i['x']
        y = t_i['y']

        vel = pd.DataFrame(np.nan, index=t_i.index.values[1:], columns=[particleID])
        acc_vel = pd.DataFrame(np.nan, index=t_i.index.values[2:], columns=[particleID])

        for frame in x.index.values[1:]:
            d = ((x.loc[frame] - x.loc[frame - 1]) ** 2 + (y.loc[frame] - y.loc[frame - 1]) ** 2) ** 0.5
            vel.loc[frame] = d * frames_second / pixels_micron
            if frame > x.index.values[1]:
                acc_vel.loc[frame] = (vel.loc[frame] - vel.loc[frame-1]) * frames_second

        return vel, acc_vel


def eval_fitness_function(t1, frames_second, pixels_micron, training_data, bounds, parameters):
    frames_av, smooth, acceleration_threshold = transform_parameters(parameters, bounds)
    change_dir_vector = []
    error_vector = []
    # make sure that frames_av is not zero. If so, set to 1
    frames_av = frames_av if frames_av != 0 else 1

    tic = time.time()
    for particle in training_data.index.values:

        # calculate velocity
        vel, acc = calculate_vel(t1, particle, frames_second, pixels_micron)

        # smooth velocity vector
        if smooth != 0 and frames_av != 0:
            av_vel = calculate_average_vel(vel, threshold=frames_av)
            for x in range(0, smooth - 1):
                av_vel = calculate_average_vel(av_vel, threshold=frames_av)
        else:
            av_vel = vel

        # calculate acceleration
        av_acc = calculate_av_acc(frames_second, av_vel)

        # get number of change of direction, duration of trajectory and frequency of change of direction
        a, b, c = get_peaks_data(t1, frames_second, abs(av_acc), acceleration_threshold)

        change_dir_vector.append(a)
        error_vector.append((a - training_data.loc[particle]['real_chng_dir']) ** 2)

    training_data['estimated_chng_dir'] = change_dir_vector
    training_data['delta_sqr'] = error_vector
    total_error = training_data['delta_sqr'].sum()
    toc = time.time()

    return total_error, toc - tic

#load data from ExcelFile
t1_ = pd.read_excel('deap_excel_data.xlsx', sheet_name=0, index_col=0)
training_data_ = pd.read_excel('deap_excel_data.xlsx', sheet_name=1, index_col=0)
video_properties = pd.read_excel('deap_excel_data.xlsx', sheet_name=2, index_col=0)

frames_second_ = video_properties.loc['frames_second']['Value']
pixels_micron_ = video_properties.loc['pixels_micron']['Value']
obj1 = video_properties.loc['obj1']['Value']
obj2 = video_properties.loc['obj2']['Value']

bounds_ = [video_properties.loc['Frames']['Value'],
          video_properties.loc['Smooth']['Value'],
          video_properties.loc['Acceleration']['Value']]

# Individual definition.
IND_SIZE = 3
if 'FitnessMin' not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-obj1, -obj2))
else:
    del creator.FitnessMin
    creator.create("FitnessMin", base.Fitness, weights=(-obj1,
                                                        -obj2))
if 'Individual' not in dir(creator):
    creator.create("Individual", list, fitness=creator.FitnessMin)
else:
    del creator.Individual
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("attr_int", random.randint, 1, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=IND_SIZE)
toolbox.register("map", futures.map)

toolbox.register("evaluate", eval_fitness_function, t1_, frames_second_, pixels_micron_, training_data_, bounds_)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutGaussian, mu=1.0, sigma=0.5, indpb=0.5)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=50)  # # tools.selNSGA2 # tools.selTournament, tournsize=50
print("selTournament")


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

    bounds_values = pd.read_excel('deap_excel_data.xlsx', sheet_name=2, index_col=0)
    bound_val = [bounds_values.loc['Frames']['Value'],
                 bounds_values.loc['Smooth']['Value'],
                 bounds_values.loc['Acceleration']['Value']]

    frames_av, smooth, acceleration_threshold = transform_parameters(halloffame[0], bound_val)

    #export results to excel file
    columns = ['Value']
    index = ['frames_av', 'smooth', 'acceleration_threshold']
    values = [frames_av, smooth, acceleration_threshold]
    results_deap = pd.DataFrame(values, index=index, columns=columns)
    results_deap.index.name = 'Parameter'

    index = ["1", "2", "3"]
    values = [halloffame[0][0], halloffame[0][1], halloffame[0][2]]
    results_halloffame = pd.DataFrame(values, index=index, columns=columns)
    results_halloffame.index.name = 'Parameter'

    writer = pd.ExcelWriter('results_deap.xlsx')
    results_deap.to_excel(writer, 'results_deap')
    results_halloffame.to_excel(writer, 'halloffame')
    writer.save()

if __name__ == "__main__":
    #print(sys.argv)
    main(sys.argv[1:])
