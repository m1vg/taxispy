from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import pims
import trackpy as tp
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from taxispy.detect_peaks import detect_peaks # Used by local self.get_peaks
from . import analysis_utils # Import the new module
import math
import random
import time
import subprocess
import platform

from io import BytesIO
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
from IPython.display import display

from deap import base, creator, tools, algorithms

__author__ = "Miguel A. Valderrama-Gomez, https://github.com/m1vg"
__version__ = "0.1.6.3"
__license__ = "MIT"

mpl.rc('image', cmap='gray')

Box = widgets.Box
VBox = widgets.VBox
Accordion = widgets.Accordion
Text = widgets.Text
IntText = widgets.IntText
FloatText = widgets.FloatText
Button = widgets.Button
Toggle = widgets.ToggleButton
HTML = widgets.HTML
IntSlider = widgets.IntSlider
Range = widgets.IntRangeSlider
FloatRange = widgets.FloatRangeSlider
Image = widgets.Image
HBox = widgets.HBox
Dropdown = widgets.Dropdown
Label = widgets.Label
Checkbox = widgets.Checkbox

# initialize the user interface.


class UserInterface(object):

    def __init__(self):

        self.path = []
        self.frames_second = []
        self.pixels_micron = []
        self.fig_size = [5, 5]
        self.frames =[]
        self.f = []
        self.t_i = []
        self.slider = None
        self.play = None
        self.frame_range = None
        self.vel = None
        self.angular = None
        self.ensemble_particle_id_list = None
        self.counter = 0
        self.interactive_ranges = False
        self.vel_ensemble = None
        self.angular_ensemble = None
        self.trajectories_dic = None
        self.cut_button = None
        self.max_displacement = None

        self.ax1_ind = None
        self.ax2_ind = None
        self.ax3_ind = None
        self.ax4_ind = None
        self.fig_individual = None

        self.acc_vel_ensemble = None

        self.av_vel_ensemble = None
        self.av_vel = None
        self.av_angular = None
        self.av_acc_vel = None
        self.peaks_table = None

        # options controlling behaviour of genetic algorithm
        self.generations = None
        self.population = None
        self.parallel = None
        self.optimal_objective_function = None
        self.weights_obj1 = None
        self.weights_obj2 = None
        self.individuals_bounds = None

        # options controlling the behavior of adaptation curve.
        self.single_trajectories = False
        self.adaptation_curve_data = None
        self.adaptation_curve_smooth_data = None
        self.show_id = False

        self.displacement_3_points = None
        self.max_displacement_3_points = None

        self.acc_vel = None
        self.acc_angular = None

        self.optimal_parameter_set = [4, 3, 10] # Default values

        # First, create four boxes, each for one of the four sections.
        self.box0 = VBox()
        self.box1 = VBox()
        self.box2 = VBox()
        self.box3 = VBox()
        self.box3_1 = VBox()
        self.box4 = VBox()
        self.box5 = VBox()
        self.box6 = VBox()

        self.peak_frame = None
        self.peak_values = None
        self.peak_height = None

        self.excel_unfiltered = None
        self.excel_filtered = None

        self.lock1 = True

        # Now, create accordion
        self.interface = Accordion(children=[self.box1, self.box2, self.box3, self.box3_1, self.box4, self.box5])
        title_list = ['File', 'Feature Identification', 'Trajectories',
                      'Visualization', 'Parameter Determination', 'Tumbling Frequencies']

        for idx, val in enumerate(title_list):
            self.interface.set_title(idx, val)

        self.populate_first_window()
        self.populate_second_window()
        self.populate_third_window()
        self.populate_fourth_window()
        self.populate_fifth_window()

    def populate_first_window(self):
        # ###################### now lets start constructing the first box.
        legend = HTML(value='<style>div.a {line-height: normal;}</style>''<div class="a">'
                            'Introduce the location of the folder containing frames to be analyzed in the '
                            '<b>Path</b> field. Then, click the <b>Load Frames</b> button. If necessary, use the '
                            '<b>Frames</b> slider to adjust the range of frames to be analyzed, '
                            'then click on <b>Cut Frames</b>.'
                            ' Adjust numerical values for <b>Frames/s</b> and <b>Pixels/Micron</b> if necessary.'
                            'Then, click on <b>Process Data</b>.  '
                            '<br><br></div>')
        path = Text(description='Path',
                    placeholder='/Users/Figures ..',
                    value='/Documents/'
                    )
        frames_second = FloatText(description='Frames/s',
                                  value=20.3)
        pixels_micron = FloatText(description='Pixels/Micron',
                                  value=6.0)
        load_data = Button(description='Process Data')
        load_data.path = path
        load_data.frames_second = frames_second
        load_data.pixels_micron = pixels_micron
        self.pixels_micron = pixels_micron
        self.frames_second = frames_second
        load_data.on_click(self.load_data_function)
        frame_segment = Range(description='Frames', min=0, max=1000, step=1, value=[0, 100])
        load_button = Button(description='Load Frames')
        load_button.path = path
        self.cut_button = Button(description='Cut Frames', disabled=True)
        self.cut_button.on_click(self.cut_frames)
        load_button.on_click(self.update_frame_segment)
        self.box1.children = [legend, path, load_button, frame_segment, self.cut_button,
                              frames_second, pixels_micron, load_data]

    def populate_second_window(self):
        # ####################### now let's construct second box
        legend0 = HTML(value='<style>div.a {line-height: normal;}</style>'
                             '<div class="a">Please select adequate values for the following parameters:<br><br></div>')

        legend1 = HTML(value='<style>div.a {line-height: normal;}</style>'
                             '<div class="a"> <br><b>Diameter</b> is given in pixels and its value should be odd. It '
                             'refers to the diameter of the particles to be identified by the software. '
                             ' <b>Min. Mass</b> '
                             'refers to the minimal mass (brightness) particles should have in order to be considered. '
                             '<b>Diameter</b> and <b>Min. Mass</b> are related.  <b>Invert</b> '
                             'refers to the color pattern. Use if cells are represented by black objects in original '
                             'raw frames. '
                             'Adequate values for minimal mass can be extracted from the histograms below. '
                             'The first, an intermediate, and the last frames are characterized '
                             'by each of the three columns shown below. Histograms are for the mass of the particles,'
                             ' blue circles are cells identified by '
                             'the software. If cells (bright objects) are not identified properly, adjust '
                             'parameter values. To continue, click on <b>Calculate Trajectories</b>: <br><br>'
                             '</div>')
        diameter = IntSlider(description='Diameter', value=25, min=1, max=99, step=2, continuous_update=False)
        diameter.observe(handler=self.update_hist_frames, names='value')
        self.diameter = diameter
        invert = Toggle(value=True,
                        description='Invert?'
                        )
        invert.observe(handler=self.update_hist_frames, names='value')
        self.invert = invert
        min_mass = IntSlider(description='Min. Mass', value=2000, min=0, max=5000, continuous_update=False)
        self.min_mass = min_mass
        self.min_mass.observe(handler=self.update_hist_frames, names='value')
        self.mass_histogram_box = Box()
        self.frames_container = Box()
        controllers = HBox()
        controllers.children = [diameter, min_mass, invert]
        button_calculate_trajectories = Button(description='Calculate Trajectories')
        button_calculate_trajectories.on_click(self.refresh_trajectories_ensemble)
        self.box2.children = [legend0, controllers, legend1, button_calculate_trajectories,
                              VBox([self.frames_container, self.mass_histogram_box])]

    def populate_third_window(self):
        # ####################### now let's construct third box
        legend4 = HTML(value='<style>div.a {line-height: normal;}</style>'
                             '<div class="a">The trajectories shown below are calculated using following parameters:'
                             ' </div>'
                       )
        legend4_1 = HTML(value ='<style>div.a {line-height: normal;}</style>'
                                '<div class="a">'
                                '<b>Max. Disp.</b> refers to the maximum displacement (in pixels) '
                                'allowed for a cell to move between frames. <b> Min. # Frms</b> refers to '
                                'the minimum number of frames that a trajectory should have to be considered. '
                                ' Please change values as required and click on <b>Recalculate</b> '
                                'to refresh results. '
                                'The number of trajectories shown in the plot on the right panel can be '
                                'reduced by increasing the displacement threshold (<b>Disp. Thrshld.</b>). '
                                'This threshold can be set to values '
                                'between 0 and 100% of the maximum displacement of all particles. '
                                'Trajectories shown exhibit a displacement that equals or surpasses the threshold set. '
                                'Alternatively, trajectories can be filtered by adjusting the frame range '
                                '(<b>Frame Rng</b>). </div>'
                        )

        legend5 = HTML(value='')
        max_displacement = IntSlider(value=self.diameter.value, description='Max. Disp.', min=0,
                                     max=self.diameter.value*5, continuous_update=False, step=1)
        self.max_displacement = max_displacement
        memory = IntSlider(value=0, description='Memory', min=0, max=0, continuous_update=False, step=1)
        self.memory = memory
        number_frames = IntSlider(value=20, description='Min. # Frms', min=0, max=40, step=1)
        self.number_frames = number_frames
        self.box_trajectories_ensemble1 = Box()
        self.box_trajectories_ensemble2 = Box()
        self.box_trajectories_ensemble = HBox(children=[self.box_trajectories_ensemble1, self.box_trajectories_ensemble2])
        controllers2 = VBox()
        controllers2.children = [max_displacement, self.number_frames]
        controllers3 = HBox()
        controller_displ = IntSlider(value=0, description='Disp. Thrshld', min=0,
                                     max=100, continuous_update=False, step=1)
        controller_time_frame = widgets.IntRangeSlider(value=[0, 10], min=0, max=10.0, step=1,
                                                       description='Frame Rng:', disabled=False,
                                                       continuous_update=False, orientation='horizontal', readout=True)
        controller_displ.observe(handler=self.filter_initial_trajectories, type='change', names='value')
        controllers3.children = [controller_displ, controller_time_frame]
        recalculate = Button(description='Recalculate')
        recalculate.on_click(self.recalculate_link)
        button_box =HBox(children=[recalculate])
        self.legend6 = HTML()
        self.box3.controller_time_frame = controller_time_frame
        self.box3.controller_displ = controller_displ
        self.box3.children = [legend4, controllers2, legend4_1, controllers3, self.box_trajectories_ensemble,
                              self.legend6, button_box, legend5]

    def populate_fourth_window(self):
        # ####################### now let's construct 3.1 Box. Visualization of a certain particle.
        self.legend3_1 = HTML(value =   '<style>div.a {line-height: normal;}</style>'
                                        '<div class="a">'
                                        'Cell trajectories identified by the software can be visualized in this window.'
                                        ' Select a trajectory from the drop-down menu and press the play button.'
                                        '<br /><br /></div>'
                              )
        self.trajectories_menu = Dropdown(description='Trajectory')
        self.trajectories_menu.observe(handler=self.update_video_parameters, type='change', names='value')
        self.video_wid = widgets.Image()


        # ####################### now let's construct fourth box
        ensemble = HTML('<b>Automatic Parameter Identification Using a Genetic Algorithm</b>')
        description = HTML('<style>div.a {line-height: normal;}</style><div class="a">'
                           'In this section, key parameters for event identification, i.e., '
                           '# Frames, # Smooth, and Acc. Thrhld, can be automatically identified using '
                           'an optimization routine. Key parameters are identified by minimizing the difference'
                           ' between the estimated and the real number of change of direction for a given set of '
                           ' trajectories. To populate the training set, first provide the number of trajectories '
                           'by adjusting the <b> # Trajectories </b> slider, then click on <b>Populate</b>. A randomly '
                           'selected training set will appear. Update this list by providing the trajectory ID and its '
                           'observed number of change of direction. Alternatively, provide the name of an Excel '
                           'file containing two columns, one for the trajectory ID and one for its respective change '
                           'of direction. The headers of these columns should be "Trajectory" and "Tumbles", '
                           'respectively. Once the training set has been loaded from an excel file or manually typed, '
                           'click on <b>Estimate Parameters</b>. Please note that this step is computationally intensive'
                           ' and might take several minutes to complete. After the optimization routine is done, '
                           'the button <b>Show Parameters</b> will appear and you can continue to the '
                           '<b>Tumbling Frequencies</b> tab. '
                           '<br /><br /></div>')
        individual = HTML('<b>Analysis of Individual Trajectories</b>')
        description_individual = HTML('Select one trajectory from the list to generate velocity plots.'
                                      ' Time can be adjusted by changing <b>Time Range</b>.')

        self.individual_metrix_box = Box()
        self.individual_controllers_box = VBox()

        training_controller_box = VBox([HBox([IntSlider(min=0, max=10,value='10', description='# Trajectories:'),
                                        Button(description='Populate')]),

                                        HBox([Text(description='File:', placeholder='Enter Excel File (.xlsx)'),
                                              Button(description='Load')]),
                                        ])

        training_controller_box.children[0].children[1].on_click(self.populate_training_set)
        training_controller_box.children[1].children[1].on_click(self.load_training_set)

        training_set = VBox()
        estimate_button = Button(description='Estimate Parameters', disabled=True)
        estimate_button.on_click(self.prepare_genetic_algorithm)
        optimal_parameters_box = VBox()
        genetic_algorithm_controller = VBox()

        self.box4.children = [individual,                           # 0
                              description_individual,               # 1
                              self.individual_controllers_box,      # 2
                              self.individual_metrix_box,           # 3
                              ensemble,                             # 4
                              description,                          # 5
                              training_controller_box,              # 6
                              training_set,                         # 7
                              genetic_algorithm_controller,         # 8
                              estimate_button,                      # 9
                              optimal_parameters_box                # 10
                              ]

    def populate_fifth_window(self):
        # ####################### now let's construct fifth box
        legend6 = HTML('Set parameters for data smoothing and event identification:')
        legend7 = HTML('<style>div.a {line-height: normal;}</style>''<div class="a">'
                       '<br />Now, set thresholds to filter trajectories with anomalous behavior. '
                       ' Use the displacement threshold to eliminate stuck cells exhibiting '
                       'high velocity. A threshold for the maximum number of change of directions (Max Chng Dir) '
                       'can be used to eliminate trajectories with excessive number of turns.<br /><br /></div>')
        legend7_1 = HTML('<style>div.a {line-height: normal;}</style>''<div class="a">'
                         '<br />In order to calculate adaptation curves, set a value for time intervals in seconds '
                         '- T. int. (s) -. To calculate the adaptation time, set a threshold value for the frequency '
                         'of change of direction (Chg. Dir.)<br /><br /></div>')
        lin_vel_threshold = widgets.BoundedIntText(value=4, min=0, max=100, step=1,
                                                   description='Velocity', disabled=False, continuous_update=True)
        acc_threshold = widgets.BoundedIntText(value=10, min=0, max=1000, step=1,
                                               description='Acceleration', disabled=False, continuous_update=False)
        disp_threshold = IntSlider(value=10, description='Dsplcmt, %', min=0,
                                   max=100, continuous_update=False, step=1)
        turns_threshold = widgets.BoundedIntText(value=10, min=0, max=100, step=1, description='Max Chng Dir',
                                                 disabled=False,continuous_update=True)
        frames_average = widgets.BoundedIntText(value=4, min=0, max=10, step=1,
                                                description='# Frames', disabled=False, continuous_update=False)
        smooth_cycles = widgets.BoundedIntText(value=3, min=0, max=10, step=1, description='# Smooth', disabled=False,
                                               continuous_update=False)
        time_interval = widgets.BoundedIntText(value=1, min=1, max=10, step=1,
                                               description='T. Int. (s)', disabled=False, continuous_update=True)
        change_dir_threshold = widgets.BoundedFloatText(value=0.45, min=0, max=2, step=0.05,
                                                        description='Chg. Dir. (1/s)', disabled=False,
                                                        continuous_update=True)
        frame_ranges = widgets.IntRangeSlider(value=[0, 10], min=0, max=10.0, step=1,
                                              description='Frame Rng:', disabled=False, continuous_update=False,
                                              orientation='horizontal', readout=True)

        b_calculate = Button(description='Calculate')
        b_calculate.on_click(self.calculate_ensemble)
        results = VBox()
        results_string = HTML()
        options_adaptation_curve = VBox()
        data_adaptation_curve = VBox()
        b_report = Button(description='Report', disabled=True)
        b_report.on_click(self.generate_report)
        self.box5.acceleration = acc_threshold
        self.box5.lin_vel_threshold = lin_vel_threshold
        self.box5.children = [legend6,                                                      # 0
                              frames_average,                                               # 1
                              smooth_cycles,                                                # 2
                              acc_threshold,                                                # 3
                              legend7,                                                      # 4
                              lin_vel_threshold,                                            # 5
                              HBox([disp_threshold, turns_threshold, frame_ranges]),        # 6
                              legend7_1,                                                    # 7
                              time_interval,                                                # 8
                              change_dir_threshold,                                         # 9
                              b_calculate,                                                  # 10
                              results,                                                      # 11
                              results_string,                                               # 12
                              options_adaptation_curve,                                     # 13
                              data_adaptation_curve,                                        # 14
                              b_report]                                                     # 15
        self.box5.frame_ranges = frame_ranges

        # ####################### now let's construct sixth box
        legend8 = HTML('Adaptation times can be calculated in this window. Required parameters are the same as '
                       'for the Ensemble Analysis window. Note that in order for a trajectory to be considered, '
                       'it must be on focus for a certain number of frames. This parameter is defined in the window'
                       ' <b>Trajectories<b> by the value of # Frames. The same is true for the parameter Max. Disp.'
                       'and all parameters from the window <b>Feature Identification<b>')
        legend9 = HTML('First set parameters for data smoothing:')
        legend10 = HTML('Now, set parameters for event identification. Then click <b>Calculate</b>')
        b_calculate2 = Button(description='Calculate')
        b_calculate2.on_click(self.calculate_adaptation_time)
        results2 = VBox()
        results_string2 = HTML()
        time_interval = widgets.BoundedFloatText(value=5, min=0, max=500, step=1,description='T. Int. (s)',
                                                 disabled=False, continuous_update=False)
        lin_vel_threshold2 = widgets.BoundedIntText(value=12, min=0, max=100, step=1,
                                                    description='Velocity', disabled=False, continuous_update=True)

        self.box6.children = [legend8,              # 0
                              legend9,              # 1
                              frames_average,       # 2
                              smooth_cycles,        # 3
                              legend10,             # 4
                              lin_vel_threshold2,   # 5
                              acc_threshold,        # 6
                              time_interval,        # 7
                              b_calculate2,         # 8
                              results2,             # 9
                              results_string2,      # 10
                              ]

    def load_data_function(self, b):
        self.box3.controller_time_frame.observe(handler=self.filter_initial_trajectories, type='change', names='value')
        # update max value of time interval for adaptation curve calculation
        self.box5.children[8].max = len(self.frames)/self.frames_second.value # updated. it was [7]
        # get number of frames and micron/pixel
        self.pixels_micron = b.pixels_micron
        self.frames_second = b.frames_second
        # this function needs to do following things:
        # load frames
        if len(self.frames) == 0:
            self.frames = pims.ImageSequence(b.path.value+'/*.jpg', as_grey=True)
        # call function that plots three frames
        self.populate_frames()
        # generate histogram of mass distribution and place it in self.mass_histogram_box
        self.refresh_histogram()
        # open next window
        self.interface.selected_index = 1
        # Generate image for frame 0
        y = mpl.pyplot
        a = y.imshow(self.frames[0])
        y.close()
        buf = BytesIO()
        canvas = FigureCanvasAgg(a.figure)
        canvas.print_png(buf)
        data = buf.getvalue()
        self.video_wid.value = data

    def refresh_histogram(self):

        # identify frames
        frames = [0, round(len(self.frames)/2), len(self.frames)-1]
        children = [Image(value=self.get_hist_data(self.frames[element])) for element in frames]
        self.mass_histogram_box.children = children

        # new mass value is b['new']
        # create histogram and place in box self.mass_histogram_box

    def refresh_trajectories_ensemble(self, b):
        # observe controller
        # Generate trajectories plot and set as children of self.box_trajectories_ensemble
        self.f = tp.batch(self.frames[:],
                          self.diameter.value,
                          minmass=self.min_mass.value,
                          invert=self.invert.value,
                          engine='numba',
                          processes='auto')
        self.generate_link(self.f)
        display(self.interface)
        self.number_frames.max = len(self.frames)-1
        self.interface.selected_index = 2
        # Modify widget 'Charactrerization'
        self.update_characterization_widget()

    def recalculate_link(self,b):
        self.generate_link(self.f)
        display(self.interface)

    def generate_link(self, f):
        self.t = tp.link_df(f, self.max_displacement.value, memory=self.memory.value)  # maximum displacement in pixels.
        self.t1 = tp.filter_stubs(self.t, self.number_frames.value)
        self.legend6.value = '<style>div.a {line-height: normal;}</style>''<div class="a"> Showing ' + \
                             str(self.t1['particle'].nunique()) + ' trajectories out of ' + \
                             str(self.t['particle'].nunique()) + ' total trajectories.' + ' </div>'
        fig_size = [7, 7]
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        yfig = tp.plot_traj(self.t1, ax=ax)
        buf = BytesIO()
        canvas = FigureCanvasAgg(yfig.figure)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        plt.close(ax.figure)
        self.box_trajectories_ensemble1.children = [Image(value=data_fig)]
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        yfig = tp.plot_traj(self.t1, ax=ax)

        # generate a new data frame containing X positions for each particle

        x = self.t1.set_index(['frame', 'particle'])['x'].unstack()
        y = self.t1.set_index(['frame', 'particle'])['y'].unstack()
        id_particles = x.columns.values
        self.trajectories_menu.options = id_particles
        self.current_ids = id_particles
        self.trajectories_menu.value = id_particles[-1]
        #update .options trait of dropdown Trajectory # of the individual trajectories in characterization widget
        self.update_characterization_widget()
        counter = 0
        for particle in id_particles:
            if counter < 200:
                #get x and y position
                x_text = x[np.isfinite(x[particle])][particle].iloc[0]
                y_text = y[np.isfinite(y[particle])][particle].iloc[0]
                #plot ID
                plt.text(x_text, y_text, str(particle), fontsize=10)
                counter += 1
            else:
                break

        buf = BytesIO()
        canvas = FigureCanvasAgg(yfig.figure)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        plt.close(ax.figure)
        self.box_trajectories_ensemble2.children = [Image(value=data_fig)]

    def populate_frames(self):
        # identify frames
        frames = [0, round(len(self.frames)/2), len(self.frames)-1]
        children = [Image(value=self.get_fig_data(self.frames[element])) for element in frames]
        self.frames_container.children = children

    def get_fig_data(self, data):
        # this scripts generate figure from frame data and return string that can be printed using the Figure widget.
        # use preset parameters to circle cells.
        f = tp.locate(data, self.diameter.value, minmass=self.min_mass.value, invert=self.invert.value)  # frame number, diameter of particle
        plt.figure(figsize=[5, 4])
        ax = plt.gca()
        ax.set(xlabel='y, [px]', ylabel='x, [px] ')
        y = tp.annotate(f, data, invert=self.invert.value, color='blue', ax=ax) # modify the function 'annotate so that I dont get output.'
        buf = BytesIO()
        canvas = FigureCanvasAgg(y.figure)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        plt.close(ax.figure)
        return data_fig

    def get_hist_data(self, data):

        plt.figure(figsize=[5, 4])
        ax = plt.gca()
        f = tp.locate(data, self.diameter.value, minmass=self.min_mass.value, invert=self.invert.value)  # frame number, size of particle
        ax.hist(f['mass'], bins=20)
        # Optionally, label the axes.
        ax.set(xlabel='mass', ylabel='count')

        buf = BytesIO()
        canvas = FigureCanvasAgg(ax.figure)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        plt.close(ax.figure)
        return data_fig

    def update_hist_frames(self, b):

        self.refresh_histogram()
        self.populate_frames()

    def update_video_parameters(self, b):
        self.slider = None
        self.play = None
        # this function gets called when a certain particle is selected. i.e, when the drop-down menu
        # self.trajectories_menu changes its trait value.
        # Generate matrix specific for one particle
        self.t_i = self.t1[self.t1['particle'] == b['new']]

        # update self.video_wid.value with the first image.
        if len(self.t_i) != 0:
            first_frame = self.t_i['frame'].iloc[0]
            plt.figure(figsize=[6, 6])
            ax = plt.gca()
            ax.set(xlabel='x, [px]', ylabel='y, [px]')
            y = tp.annotate(self.t_i[self.t_i['frame'] == first_frame], self.frames[first_frame],
                            color='blue', invert=False, ax=ax);
            buf = BytesIO()
            canvas = FigureCanvasAgg(y.figure)
            canvas.print_png(buf)
            data_fig = buf.getvalue()
            plt.close(ax.figure)
            self.video_wid.value = data_fig

        #update values of self.play & self.slider.
        self.play = widgets.Play(
            value=0,
            min=0,
            max=len(self.t_i['frame']),
            step=1,
            description="Press play",
            disabled=False)
        self.slider = widgets.IntSlider(continuous_update=True,
                                        value=0, min=0, max=len(self.t_i['frame']),
                                        description='Frame #')
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        self.slider.observe(handler=self.update_video, type='change', names='value')
        self.trajectories_menu.observe(handler=self.update_video_parameters, type='change', names='value')
        single_trajectory = self.get_single_trajectory(self.t_i)
        self.box3_1.children = [self.legend3_1, widgets.HBox([self.trajectories_menu, self.play, self.slider]),
                                HBox([Box([self.video_wid]), Box([single_trajectory])])]
    def update_video(self, b):
        counter = b['new'] # contains iloc of self.t_i

        if counter < len(self.t_i):
            frame_id = self.t_i['frame'].iloc[counter]
            plt.figure(figsize=[6, 6])
            ax = plt.gca()
            ax.set(xlabel='x, [px]', ylabel='y, [px]')
            y = tp.annotate(self.t_i[self.t_i['frame'] == frame_id], self.frames[frame_id],
                        color='blue', invert=False, ax=ax);
            plt.text(100, 100, str(round(frame_id/self.frames_second.value, 3)) + ' s', color='white')

            buf = BytesIO()
            canvas = FigureCanvasAgg(y.figure)
            canvas.print_png(buf)
            data_fig = buf.getvalue()
            plt.close(ax.figure)
            self.video_wid.value = data_fig

    def update_characterization_widget(self):
        # current ids are in self.current_ids
        # update ensemble Box. Target box: self.ensemble_controllers_box.children
        # update individual box. Target: self.individual_controllers_box.children

        self.trajectories_id = Dropdown(description='Trajectory #', options=self.current_ids)
        self.trajectories_id.observe(handler=self.update_frame_range, type='change', names='value')
        self.trajectories_id.value = self.current_ids[-1]
        self.box4.children[6].children[0].children[0].max = len(self.current_ids)

    def update_frame_range(self, b):
        # b['new'] contains the ID of the particle.

        t_i = self.t1[self.t1['particle'] == b['new']]
        min_value = t_i['frame'].iloc[0]/self.frames_second.value if t_i['frame'].iloc[0]/self.frames_second.value != 0 \
                    else 1/self.frames_second.value
        max_value = t_i['frame'].iloc[-1]/self.frames_second.value

        frame_range = FloatRange(value=[min_value, max_value],
                            min=min_value,
                            max=max_value,
                            step=1/self.frames_second.value,
                            description='Time Range',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            readout_format='.2f')

        threshold_mean = widgets.BoundedIntText(
            value=self.optimal_parameter_set[0],
            min=0,
            max=len(self.frames),
            step=1,
            description='# Frames',
            disabled=False,
            continuous_update=True)

        smooth_cycles = widgets.BoundedIntText(
            value=self.optimal_parameter_set[1],
            min=0,
            max=10,
            step=1,
            description='# Smooth',
            disabled=False,
            continuous_update=True)

        acceleration_threshold = widgets.BoundedIntText(
            value=self.optimal_parameter_set[2],
            min=0,
            max=1000,
            step=1,
            description='Acc. Thrhld',
            disabled=False,
            continuous_update=False)

        self.individual_controllers_box.children = [self.trajectories_id,
                                                    frame_range,
                                                    threshold_mean,
                                                    smooth_cycles,
                                                    acceleration_threshold]

        frame_range.observe(handler=self.print_individual_characterization, type='change', names='value')

        if self.interface.selected_index == 4:
            particle_id = b['new'] 
            
            # Prepare DataFrame for the selected particle, indexed by 'frame'
            # This is the expected input format for analysis_utils.calculate_vel
            if self.t1 is not None and not self.t1.empty:
                t1_particle_df = self.t1[self.t1['particle'] == particle_id]
                if not t1_particle_df.empty:
                    t1_particle_df_indexed = t1_particle_df.set_index('frame')
                else: # Particle ID not found or t1 empty for this particle
                    t1_particle_df_indexed = pd.DataFrame() # Empty DF
            else: # self.t1 not loaded
                t1_particle_df_indexed = pd.DataFrame()

            self.vel, self.acc_vel = analysis_utils.calculate_vel(
                t1_particle_df_indexed, self.frames_second.value, self.pixels_micron.value
            )
            
            smooth_param = int(self.optimal_parameter_set[1])
            n_frames_param = int(self.optimal_parameter_set[0])

            self.av_vel = self.vel.copy() 
            if smooth_param > 0 and n_frames_param > 0 and not self.av_vel.empty:
                for _ in range(smooth_param):
                    self.av_vel = analysis_utils.calculate_average_vel(self.av_vel, threshold=n_frames_param)
            
            if not self.av_vel.empty:
                self.av_acc_vel = analysis_utils.calculate_av_acc(self.av_vel, self.frames_second.value)
            else:
                self.av_acc_vel = pd.DataFrame() # Empty DF

            c = {'new': [min_value, max_value]}
            
            # self.get_peaks is a local method for UI update of self.peaks_table
            if not self.av_acc_vel.empty and len(self.av_acc_vel.columns) > 0:
                 # Pass the actual series to get_peaks
                 self.get_peaks(abs(self.av_acc_vel[self.av_acc_vel.columns[0]])) 
            else: 
                 self.peaks_table = pd.DataFrame() 
                 self.peak_height = self.individual_controllers_box.children[4].value 
            self.print_individual_characterization(c)

            threshold_mean.time_range = c
            threshold_mean.particle = particle

            smooth_cycles.time_range = c
            smooth_cycles.particle = particle

            acceleration_threshold.time_range = c
            acceleration_threshold.particle = particle

            threshold_mean.observe(handler=self.update_average_vel, type='change', names='value')
            smooth_cycles.observe(handler=self.update_average_vel, type='change', names='value')
            acceleration_threshold.observe(handler=self.update_average_vel, type='change', names='value')

    def update_average_vel(self, b): 
        particle_id = b['owner'].particle 
        
        n_frames_param = self.individual_controllers_box.children[2].value
        smooth_param = self.individual_controllers_box.children[3].value
        acc_thresh_param_mph = self.individual_controllers_box.children[4].value

        if self.t1 is None or self.t1.empty:
            # Handle cases where self.t1 might not be loaded
            self.vel, self.acc_vel, self.av_vel, self.av_acc_vel = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            self.peaks_table = pd.DataFrame()
            self.peak_height = acc_thresh_param_mph
            # Update plot with empty data or a message
            time_range_val = b['owner'].time_range 
            self.print_individual_characterization(time_range_val)
            return

        t1_particle_df = self.t1[self.t1['particle'] == particle_id]
        if t1_particle_df.empty: # Particle not found
             self.vel, self.acc_vel, self.av_vel, self.av_acc_vel = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
             self.peaks_table = pd.DataFrame()
             self.peak_height = acc_thresh_param_mph
             time_range_val = b['owner'].time_range
             self.print_individual_characterization(time_range_val)
             return
        
        t1_particle_df_indexed = t1_particle_df.set_index('frame')

        self.vel, self.acc_vel = analysis_utils.calculate_vel(
            t1_particle_df_indexed, self.frames_second.value, self.pixels_micron.value
        )

        self.av_vel = self.vel.copy()
        if smooth_param > 0 and n_frames_param > 0 and not self.av_vel.empty:
            for _ in range(smooth_param): 
                self.av_vel = analysis_utils.calculate_average_vel(self.av_vel, threshold=n_frames_param)
        
        if not self.av_vel.empty:
            self.av_acc_vel = analysis_utils.calculate_av_acc(self.av_vel, self.frames_second.value)
        else:
            self.av_acc_vel = pd.DataFrame()
        
        if not self.av_acc_vel.empty and len(self.av_acc_vel.columns) > 0:
            # Pass the actual series to get_peaks
            self.get_peaks(abs(self.av_acc_vel[self.av_acc_vel.columns[0]]), mph=acc_thresh_param_mph)
        else: 
            self.peaks_table = pd.DataFrame()
            self.peak_height = acc_thresh_param_mph
            
        time_range_val = b['owner'].time_range 
        self.print_individual_characterization(time_range_val)

    def print_individual_characterization(self, time_range_val_dict):

        # target: self.individual_metrix_box
        # actual trajectory is contained in self.trajectories_id.value
        # x = self.t1.set_index(['frame', 'particle'])['x'].unstack()
        # y = self.t1.set_index(['frame', 'particle'])['y'].unstack()
        # time_range_val_dict is what was 'c' before, e.g. {'new': [min_value, max_value]}
        time_frame = time_range_val_dict['new'] # Extract the list [min_time, max_time]
        min_t_val = time_frame[0] 
        max_t_val = time_frame[1]
        
        self.fig_individual, ((self.ax1_ind, self.ax2_ind),
                              (self.ax3_ind, self.ax4_ind)) = plt.subplots(2, 2,
                                                                           figsize=[15, 10],
                                                                           sharex='all')
        fs = self.frames_second.value
        if fs == 0: fs = 1 # Avoid division by zero for plotting if fs is not set

        def time_indexed_df(df, frames_per_second_val):
            if df is None or df.empty:
                return pd.DataFrame()
            # Assuming index is frame numbers for velocity/acceleration DFs
            new_idx = df.index / frames_per_second_val if frames_per_second_val !=0 else df.index
            df_time_indexed = df.copy()
            df_time_indexed.index = new_idx
            return df_time_indexed

        vel_to_plot = time_indexed_df(self.vel, fs)
        acc_vel_to_plot = time_indexed_df(self.acc_vel, fs)
        av_vel_to_plot = time_indexed_df(self.av_vel, fs)
        av_acc_vel_to_plot = time_indexed_df(self.av_acc_vel, fs)
        
        # Ensure peaks_table is also time-indexed for plotting
        # self.peaks_table index is assumed to be frame numbers from get_peaks
        peaks_table_to_plot = time_indexed_df(self.peaks_table, fs)


        # instantaneous velocities
        ax = self.ax1_ind
        if not vel_to_plot.empty:
            ax.plot(vel_to_plot.loc[min_t_val:max_t_val], color='blue')
        ax.set_ylabel('Linear Velocity, micron/s', color='blue')
        ax.tick_params('y', colors='blue')
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

        # instantaneous accelerations
        ax3 = self.ax3_ind
        if not acc_vel_to_plot.empty:
            ax3.plot(acc_vel_to_plot.loc[min_t_val:max_t_val], color='grey')
        ax3.set_ylabel('Acceleration, micron/s/s', color='grey')
        ax3.tick_params('y', colors='grey')
        ax3.grid(color='grey', linestyle='--', linewidth=0.5)

        # Average Velocities
        ax5 = self.ax2_ind
        if not av_vel_to_plot.empty:
            ax5.plot(av_vel_to_plot.loc[min_t_val:max_t_val], color='blue')
        ax5.tick_params('y', colors='blue')

        ax6 = ax5.twinx()
        if not av_acc_vel_to_plot.empty:
            # Assuming av_acc_vel_to_plot has one column, take abs of that.
            # If it can have multiple, this needs adjustment.
            col_to_abs = av_acc_vel_to_plot.columns[0] if len(av_acc_vel_to_plot.columns) > 0 else None
            if col_to_abs is not None:
                 ax6.plot(abs(av_acc_vel_to_plot[col_to_abs].loc[min_t_val:max_t_val]), color='grey', alpha=0.5)
        ax6.set_ylabel('Absolute Acceleration, micron/s/s', color='grey')
        ax6.tick_params('y', colors='grey')
        ax6.grid(color='grey', linewidth=0.5)

        if not peaks_table_to_plot.empty:
            t_peaks_in_range = peaks_table_to_plot.loc[min_t_val:max_t_val]
            if not t_peaks_in_range.empty:
                 ax6.scatter(t_peaks_in_range.index.values, t_peaks_in_range.iloc[:,0].values, marker='*', s=300, alpha=0.5, c='grey')
        
        current_mph_threshold = self.individual_controllers_box.children[4].value
        ax6.plot([min_t_val, max_t_val], [current_mph_threshold, current_mph_threshold], linewidth=2, color='black')

        # Average Accelerations
        ax7 = self.ax4_ind
        if not av_acc_vel_to_plot.empty:
            ax7.plot(av_acc_vel_to_plot.loc[min_t_val:max_t_val], color='grey')
        ax7.plot([min_t_val + (1/fs if fs > 0 else 0), max_t_val], [0, 0], color='black', linestyle='--', alpha=0.5)
        ax7.tick_params('y', colors='grey')
        ax7.grid(color='grey', linestyle='--', linewidth=0.5)

        self.ax1_ind.set_title('Instantaneous Velocities')
        self.ax2_ind.set_title('Average Velocities')
        self.ax3_ind.set_title('Acceleration (Inst. Vel)')
        self.ax3_ind.set_xlabel('Time, s')
        self.ax4_ind.set_title('Acceleration (Avg. Vel)')
        self.ax4_ind.set_xlabel('Time, s')

        data_fig = self.easy_print(self.fig_individual) 
        plt.close(self.fig_individual) 

        self.individual_metrix_box.children = [Image(value=data_fig)]

    # Removed: calculate_vel (moved to analysis_utils)
    # Removed: calculate_average_vel (moved to analysis_utils)

    def prepare_data_print_histogram(self, data, title):

        plt.figure()
        ax = plt.gca()

        if title == 'Linear Velocity' or 'Angular Velocity':
            self.refresh_particle_dictionary()
            # get a data frame and return Image data with histogram.
            # for each column in data, we first:
            # 1. get a slice of data corresponding to selected time ranges
            # 2. drop the nan values with s.dropnan
            # 3. concatenate resulting vector to main vector/list
            # 4. calculate histogram. Generate axes ax. and set title
            filtered_data = []

            for column in data.columns:
                # filter to consider only time frame selected for each column
                # eliminate dropna and get values
                min_value = self.trajectories_dic[column][0] if self.trajectories_dic[column][0] != 0 \
                    else 1/self.frames_second.value
                max_value = self.trajectories_dic[column][1]
                z = data[column].loc[min_value*self.frames_second.value:
                                     max_value*self.frames_second.value].dropna().values
                filtered_data = np.append(filtered_data, z)

            ax.hist(filtered_data)

        if title == 'Frequency of Change of Direction':
            self.refresh_particle_dictionary()
            ax.clear()
            filtered_data2 = []
            for column in data.columns:
                # filter to consider only time frame selected for each column
                # eliminate dropna and get values
                min_value = self.trajectories_dic[column][0] if self.trajectories_dic[column][0] != 0 \
                    else 1 / self.frames_second.value
                max_value = self.trajectories_dic[column][1]
                z = sum(data[column].loc[min_value*self.frames_second.value: max_value*self.frames_second.value] >
                        self.threshold_angular.value)
                time = max_value-min_value
                freq = z/time
                filtered_data2 = np.append(filtered_data2, freq)
            ax.hist(filtered_data2)
            # number of change of direction/second use self.threshold_angular

        # Extract image information from axes contained in ax and return image data.
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        buf = BytesIO()
        canvas = FigureCanvasAgg(ax.figure)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        plt.close(ax.figure)
        self.interactive_ranges = True
        return data_fig

    def refresh_particle_dictionary(self):
        trajectories_dic = dict((element[0].value, element[1].children[0].value)
                                for element in self.ensemble_particle_id_list)
        self.trajectories_dic = trajectories_dic

    def update_frame_segment(self, b):
        self.frames = pims.ImageSequence(b.path.value + '/*.jpg', as_grey=True)
        self.box1.children[3].max = len(self.frames)-1
        self.box1.children[3].value = [0, len(self.frames)-1]
        self.box1.children[3].continuous_update = False
        self.box1.children[3].observe(handler=self.reshape_frames, type='change', names='value') # frame_segment widget
        
        if self.frames and len(self.frames) > 0 : # Check if self.frames is not None and not empty
            max_frame_val = len(self.frames) - 1
            self.box3.controller_time_frame.max = max_frame_val
            self.box3.controller_time_frame.value = [0, max_frame_val if max_frame_val > 0 else 0]
            self.box5.frame_ranges.max = max_frame_val
            self.box5.frame_ranges.value = [0, max_frame_val if max_frame_val > 0 else 0]
        else: 
            self.box3.controller_time_frame.max = 0
            self.box3.controller_time_frame.value = [0,0]
            self.box5.frame_ranges.max = 0
        self.box5.frame_ranges.value = [0, len(self.frames)-1]

    def reshape_frames(self, b):
        self.cut_button.disabled = False
        ranges = b['new']
        self.cut_button.ranges = ranges

    def cut_frames(self, b):
        ranges = b.ranges
        if ranges[0] != 0:
            return
        self.frames = self.frames[ranges[0]:ranges[1]+1]
        self.box1.children[3].max = len(self.frames) - 1
        self.box1.children[3].value = [0, len(self.frames) - 1]
        self.box3.controller_time_frame.max = len(self.frames)-1
        self.box3.controller_time_frame.value = [0, len(self.frames)-1]
        self.box5.frame_ranges.max = len(self.frames)-1
            self.box5.frame_ranges.value = [0,0]


    def easy_print(self, fig_or_ax):
        """ Helper to print a Matplotlib figure or axes to a BytesIO buffer. """
        target_fig = fig_or_ax.figure if hasattr(fig_or_ax, 'figure') else fig_or_ax
        buf = BytesIO()
        canvas = FigureCanvasAgg(target_fig)
        canvas.print_png(buf)
        data_fig = buf.getvalue()
        return data_fig

    # Removed: calculate_av_acc (moved to analysis_utils)

    def get_peaks(self, vel_series_abs, mph=10): # vel_series_abs should be a pd.Series

        # vel_series_abs is the absolute acceleration series (e.g., abs(self.av_acc_vel[col_name]))
        # This method updates self.peaks_table for UI plotting.
        
        # Ensure it's a Series. If it's a DataFrame, this needs to be handled.
        # The calling context in update_average_vel should pass a Series.
        if not isinstance(vel_series_abs, pd.Series):
             # Try to convert if it's a single-column DataFrame
            if isinstance(vel_series_abs, pd.DataFrame) and len(vel_series_abs.columns) == 1:
                vel_series_abs = vel_series_abs.iloc[:, 0]
            else: # Cannot proceed
                self.peaks_table = pd.DataFrame()
                self.peak_height = mph # Store the mph used
                return

        vel_series_to_process = vel_series_abs.dropna()
        if vel_series_to_process.empty:
            self.peaks_table = pd.DataFrame()
            self.peak_height = mph
            return
            
        frames_idx = vel_series_to_process.index.values # These are original frame numbers
        values_np = vel_series_to_process.values
        
        # detect_peaks is imported from taxispy.detect_peaks
        peakind = detect_peaks(values_np, mph=mph) 
        
        peak_values_np = values_np[peakind] if peakind.size > 0 else np.array([])
        peak_frame_np = frames_idx[peakind] if peakind.size > 0 else np.array([])
        
        col_name_for_output = vel_series_to_process.name if vel_series_to_process.name is not None else 'peak_value'
        self.peaks_table = pd.DataFrame({col_name_for_output: peak_values_np}, index=peak_frame_np)
        self.peak_height = mph

    def calculate_ensemble(self, b):
        b.description = 'Calculating...'
        b.disabled = True
        self.lock1 = False
        # Target:         self.box5.children=[legend6,      #0
        #                             frames_average,       #1
        #                             smooth_cycles,        #2
        #                             legend7,              #3
        #                             lin_vel_threshold,    #4
        #                             acc_threshold,        #5
        #                             b_calculate,          #6
        #                             results,              #7
        #                              ]

        # create dataFrame
        columns = ['Velocity',          #0
                   'Acceleration',      #1
                   'Mean_vel',          #2
                   'Change_dir',        #3
                   'Duration',          #4
                   'Change_dir_sec',    #5
                   'Displacement',      #6
                   'x',                 #7
                   'y',                 #8
                   'First_Time',        #9
                   'firstFrame']        #10

        # Get indices for unique particles
        y = self.t1.set_index(['frame', 'particle'])['y'].unstack()
        id_particles = y.columns.values

        # create empty data frame that will contain all results.
        results = pd.DataFrame(np.nan, index=id_particles, columns=columns)
        results.index.name = 'Particle'

        # vel_dic, smooth_vel_dic, smooth_acc_dic are local and not used beyond this scope; can be removed.

        for particle_id in results.index.values:
            particle_data_full = self.t1[self.t1['particle'] == particle_id]
            if particle_data_full.empty or len(particle_data_full) < 2:
                results.loc[particle_id, ['Mean_vel', 'Change_dir', 'Duration', 'Change_dir_sec']] = np.nan, 0, 0, 0
                continue

            particle_data_indexed = particle_data_full.set_index('frame')

            vel_df, _ = analysis_utils.calculate_vel(
                particle_data_indexed, self.frames_second.value, self.pixels_micron.value
            )
            if vel_df.empty: 
                results.loc[particle_id, ['Mean_vel', 'Change_dir', 'Duration', 'Change_dir_sec']] = np.nan, 0, 0, 0
                continue

            smooth_param = self.box5.children[2].value 
            n_frames_param = self.box5.children[1].value 
            
            av_vel_df = vel_df.copy()
            if smooth_param > 0 and n_frames_param > 0:
                for _ in range(smooth_param):
                    av_vel_df = analysis_utils.calculate_average_vel(av_vel_df, threshold=n_frames_param)
            
            if av_vel_df.empty:
                results.loc[particle_id, ['Mean_vel', 'Change_dir', 'Duration', 'Change_dir_sec']] = np.nan, 0, 0, 0
                continue
            
            av_acc_df = analysis_utils.calculate_av_acc(av_vel_df, self.frames_second.value)
            if av_acc_df.empty:
                results.loc[particle_id, ['Mean_vel', 'Change_dir', 'Duration', 'Change_dir_sec']] = np.nan, 0, 0, 0
                continue
            
            mean_vel_val = np.nan
            if not av_vel_df.empty and len(av_vel_df.columns) > 0:
                 # Ensure we are taking mean of the correct column if multiple exist (should be one)
                 mean_vel_val = np.average(av_vel_df[av_vel_df.columns[0]].dropna().values)
            results.loc[particle_id]['Mean_vel'] = mean_vel_val

            results.loc[particle_id]['x'], \
            results.loc[particle_id]['y'], \
            results.loc[particle_id]['First_Time'] = self.get_x_y_time(particle_data_full, particle_id)

            if not av_acc_df.empty and len(av_acc_df.columns) > 0:
                abs_av_acc_series = abs(av_acc_df[av_acc_df.columns[0]])
                chng_dir, duration, chng_dir_sec = analysis_utils.get_peaks_data(
                    particle_data_full, 
                    self.frames_second.value,
                    abs_av_acc_series,
                    self.box5.acceleration.value 
                )
                results.loc[particle_id]['Change_dir'] = chng_dir
                results.loc[particle_id]['Duration'] = duration
                results.loc[particle_id]['Change_dir_sec'] = chng_dir_sec
            else:
                results.loc[particle_id, ['Change_dir', 'Duration', 'Change_dir_sec']] = 0, 0, 0
            
            results.loc[particle_id]['Displacement'], \
            results.loc[particle_id]['firstFrame'] = get_displacement(self.t1, particle_id)

        # At this point, the data frame 'results' contain data for all particles. Now we need to filter for minimum
        # linear velocity threshold, minimum Displacement and max number of change of direction.
        self.box5.lin_vel_threshold.results = results
        self.box5.children[6].children[0].results = results
        self.box5.children[6].children[1].results = results
        self.box5.frame_ranges.results = results


        percentage = self.box5.children[6].children[0].value
        filtered_results = results[results['Mean_vel'] > self.box5.lin_vel_threshold.value]
        filtered_results = filtered_results[filtered_results['Displacement'] >=
                                            results['Displacement'].max() * percentage / 100]
        filtered_results = filtered_results[filtered_results['Change_dir'] <= self.box5.children[6].children[1].value]
        self.box5.lin_vel_threshold.observe(handler=self.update_results, type='change', names='value')
        self.box5.children[6].children[0].observe(handler=self.update_results, type='change', names='value')
        self.box5.children[6].children[1].observe(handler=self.update_results, type='change', names='value')
        self.box5.frame_ranges.observe(handler=self.update_results, type='change', names='value')


        # now the next step is to print results to trajectories figure and histograms.
        self.print_ensemble_results(filtered_results)

        self.excel_unfiltered = results
        self.excel_filtered = filtered_results

        # call function to generate adaptation curve
        if self.box5.children[8].value != 0:
            self.calculate_adaptation_time_ensemble(filtered_results, self.box5.children[8], self.box5.children[9])
            # attach filtered results to calculate button.
            self.box5.children[10].results = filtered_results

            # observe changes in "change of direction". self.box5.children[8] and "time interval" self.box5.children[7]
            self.box5.children[9].observe(handler=self.update_adaptation_ensemble, names='value') # updated
            self.box5.children[8].observe(handler=self.update_adaptation_ensemble, names='value') # updated

        #populate self.box5.children[13] which shows settings used to calculate adaptation curve.
        self.display_adaptation_options()
        self.box5.children[13].children[1].children[1].results = results
        self.box5.children[13].children[1].children[1].observe(handler=self.update_results, type='change', names='value')
        self.display_adaptation_data()
        b.description = 'Calculate'
        b.disabled = False


    def display_adaptation_data(self):
        button = Button(description='Show Data')
        button.on_click(self.print_adaptation_data)
        data_table = HTML()
        tumbling_range_window = VBox()
        label1 = HTML(value='Calculate tumbling frequency (1/s) for a given time frame:')
        max_value = round(len(self.frames) / self.frames_second.value, 2)
        time_range = widgets.FloatRangeSlider(value=[0, max_value], min=0, max=max_value, step=0.1,
                                              description='T. Frame (s):', disabled=False, continuous_update=False,
                                              orientation='horizontal', readout=True, readout_format='.1f',)
        label2 = HTML()
        label2.value = 'The tumbling frequency is...'
        tumbling_range_window.children = [label1, HBox(children=[time_range, label2])]
        self.box5.children[14].children = [button, data_table, tumbling_range_window]
        time_range.observe(handler=self.calculate_tumbling_frequency_time_frame, names='value')
        self.calculate_tumbling_frequency_time_frame(time_range)

    def calculate_tumbling_frequency_time_frame(self, b):
        min_time = self.box5.children[14].children[2].children[1].children[0].value[0]
        max_time = self.box5.children[14].children[2].children[1].children[0].value[1]
        data = self.excel_filtered
        tumbling_freq, _, _, number = self.slice_data_frame(data, min_time, max_time)
        self.box5.children[14].children[2].children[1].children[1].value = 'The tumbling frequency is ' \
                                                                           + str(round(tumbling_freq,2)) + \
                                                                           ' 1/s. Number of trajectories is ' + \
                                                                           str(number)

    def print_adaptation_data(self, b):
        data = self.adaptation_curve_data
        if b.description == 'Show Data':
            s = generate_adaptation_string(data)
            self.box5.children[14].children[1].value = s
            self.box5.children[14].children[0].description = 'Hide Data'

        else:
            self.box5.children[14].children[1].value = ''
            self.box5.children[14].children[0].description = 'Show Data'

    def display_adaptation_options(self):
        string1 = 'Adaptation curves calculated using single trajectories: '
        checkbox1 = Checkbox(value=self.single_trajectories)
        string2 = 'Show ID: '
        checkbox2 = Checkbox(value=self.show_id)
        checkbox2.is_show_id_checkbox = True
        smooth_adaptation = Button(description='Smooth Adptn Crv')
        smooth_adaptation.on_click(self.display_adaptation_smooth_options)
        smooth_options = VBox()
        smooth_options.content = [widgets.BoundedIntText(description='# Points:', value=2, step=1, min=0),    # 0
                                  widgets.BoundedIntText(description='# Smooth:', value=0, step=1, min=0)     # 1
                                  ]
        smooth_adaptation.options_box = smooth_options
        self.box5.children[13].children = [HBox(children=[Label(value=string1), checkbox1]),    # 0
                                           HBox(children=[Label(value=string2), checkbox2]),    # 1
                                           smooth_adaptation,                                   # 2
                                           smooth_options,                                      # 3
                                           ]
        checkbox1.observe(handler=self.recalculate_adaptation_ensemble, names='value')

        smooth_options.content[0].observe(handler=self.smooth_adaptation_curve, names='value')
        smooth_options.content[1].observe(handler=self.smooth_adaptation_curve, names='value')

    def smooth_adaptation_curve(self, b):

        # modify self.adaptation_curve_smooth_data
        data = self.adaptation_curve_data
        n_points = self.box5.children[13].children[3].children[0].value
        n_smooth = self.box5.children[13].children[3].children[1].value
        smooth_data = pd.DataFrame(index=data.index, columns=data.columns)

        if n_smooth == 0 or n_points == 0:
            self.adaptation_curve_smooth_data = None

        if n_points != 0:
            for x in range(n_smooth):
                counter = 0
                data = self.adaptation_curve_data if x == 0 else smooth_data
                for _ in self.adaptation_curve_data.index.values:
                    if counter <= n_points:
                        smooth_data['Frequency'].iloc[counter] = np.mean(data['Frequency'].iloc[:counter+1].dropna().values)
                    else:
                        smooth_data['Frequency'].iloc[counter] = \
                            np.mean(data['Frequency'].iloc[counter - n_points + 1:counter+1].dropna().values)
                    counter += 1
            self.adaptation_curve_smooth_data = smooth_data

        # Then call the plotting functions.
        self.calculate_adaptation_time_ensemble(self.box5.children[10].results,
                                                self.box5.children[8],
                                                self.box5.children[9])
        self.update_adaptation_curve_table()

    def display_adaptation_smooth_options(self, b):

        if b.description == 'Smooth Adptn Crv':
            b.options_box.children = b.options_box.content
            b.description = 'Hide'
        else:
            b.description = 'Smooth Adptn Crv'
            b.options_box.children = []

    def update_adaptation_ensemble(self, b):
        # check that b.new is different from 0.
        #if b['new'] != 0:

        if self.adaptation_curve_smooth_data is None:
            self.calculate_adaptation_time_ensemble(self.box5.children[10].results,
                                                    self.box5.children[8],
                                                    self.box5.children[9])
            self.update_adaptation_curve_table()
        else:
            self.adaptation_curve_smooth_data = None
            self.calculate_adaptation_time_ensemble(self.box5.children[10].results,
                                                    self.box5.children[8],
                                                    self.box5.children[9])
            self.smooth_adaptation_curve(None)





    def recalculate_adaptation_ensemble(self, b):
        self.single_trajectories = b['new']
        self.calculate_adaptation_time_ensemble(self.box5.children[10].results,
                                                self.box5.children[8],
                                                self.box5.children[9])
        self.calculate_tumbling_frequency_time_frame(None)
        self.update_adaptation_curve_table()

    def update_adaptation_curve_table(self):
        if self.box5.children[14].children[0].description == 'Hide Data':
            if self.adaptation_curve_smooth_data is None:
                self.box5.children[14].children[1].value = generate_adaptation_string(self.adaptation_curve_data,
                                                                                      smooth=False)
            else:
                self.box5.children[14].children[1].value = generate_adaptation_string(self.adaptation_curve_smooth_data,
                                                                                      smooth=True)


    def calculate_adaptation_time_ensemble(self, data, time_interval, change_dir_thrshld):

        # define loop using time_interval & total time
        # slice dataFrame results to consider only particles in a certain interval
        # calculate mean, min and max and assign to data frame
        # generate plot

        time_interval = time_interval.value
        change_dir_thrshld = change_dir_thrshld.value

        # construct a dataFrame to save results.
        columns = ['Frequency',     # 0
                   'min',           # 1
                   'max',           # 2
                   'Number']        # 3

        # create empty data frame that will contain all results.
        index_array = np.arange(time_interval,
                                len(self.frames) / self.frames_second.value + time_interval,
                                time_interval)
        index_array = np.round(index_array, 3)
        results = pd.DataFrame(np.nan, index=index_array, columns=columns)
        results.index.name = 'Time'

        # The variable time_interval_frames represent the number of frames that need to be analyzed. Now we need to
        # construct a for loop that split the total number of frames into time_interval_frames
        for y in range(1, int(math.ceil(len(self.frames)/self.frames_second.value / time_interval) + 1)):
            # identify frames that belong to each group.
            initial = (y - 1) * time_interval
            final = y * time_interval
            results.loc[final]['Frequency'], \
            results.loc[final]['min'], \
            results.loc[final]['max'], \
            results.loc[final]['Number'] = self.slice_data_frame(data, initial, final)

        # plot results. --> Results are a scatter plot of frequency of change of direction vs. time interval
        ImageContainer = HBox()
        ImageContainer.children = [self.do_adaptation_plot(results, threshold=change_dir_thrshld)]
        self.adaptation_curve_data = results

        # try to put this figure next to the trajectories plot
        self.box5.children[11].children[0].children = [self.box5.children[11].image1, ImageContainer]

        # create legend.
        html_string = '<style>div.a {line-height: normal;}</style><div class="a">'
        html_string += 'Showing adaptation plot calculated using a time interval of ' \
                      + str(time_interval) + ' seconds. The parameters used for these calculations ' \
                      'were: diameter: ' + str(self.diameter.value) + '; min value for mass: ' \
                      + str(self.min_mass.value) + '; invert colors?: ' + str(self.invert.value) + \
                      '; maximum displacement: ' + str(self.max_displacement.value) + \
                      '; minimum number of frames: ' + str(self.number_frames.value) + '.'
        html_string += '<br /><br /></div>'

        self.box5.children[11].children[1].value = html_string

    def slice_data_frame(self, data, initial, final):

        lo = data[data['First_Time'] < final]
        u = lo[initial <= lo['First_Time']]

        if self.single_trajectories is True:

                frequency = np.round(np.average(u['Change_dir_sec']), decimals=2) \
                    if u['Change_dir_sec'].count()>0 else np.nan

                min_frequency = frequency - np.nanstd(u['Change_dir_sec']) / math.sqrt(u['Change_dir_sec'].count()) \
                    if u['Change_dir_sec'].count() > 0 else np.nan

                max_frequency = frequency + np.nanstd(u['Change_dir_sec']) / math.sqrt(u['Change_dir_sec'].count()) \
                    if u['Change_dir_sec'].count() > 0 else np.nan

                number = u['Change_dir_sec'].count()
        else:
                totalChangeOfDirection = np.sum(u['Change_dir'])
                totalDuration = np.sum(u['Duration'])

                frequency = np.round(totalChangeOfDirection/totalDuration, decimals=2) \
                    if u['Change_dir'].count() > 0 else np.nan

                min_frequency = np.nan
                max_frequency = np.nan
                number = u['Change_dir'].count()

        return frequency, min_frequency, max_frequency, number

    def get_x_y_time(self, t, particle):

        # particle_df is the DataFrame for the specific particle, already filtered.
        if particle_df.empty: # Should be t_i from original
            return np.nan, np.nan, np.nan 
        
        # Assuming particle_df (t_i) has 'frame' as a column
        if 'frame' not in particle_df.columns or particle_df.empty:
            return particle_df['x'].iloc[0] if 'x' in particle_df.columns and not particle_df.empty else np.nan, \
                   particle_df['y'].iloc[0] if 'y' in particle_df.columns and not particle_df.empty else np.nan, \
                   np.nan

        first_x = particle_df['x'].iloc[0]
        first_y = particle_df['y'].iloc[0]
        first_frame_val = particle_df['frame'].iloc[0]
        
        time_val = np.nan
        if self.frames_second.value and self.frames_second.value != 0:
            time_val = first_frame_val / self.frames_second.value
        return first_x, first_y, time_val


    def get_x_y(self, particle_df, particle_id_unused): # particle_id is not used here
        # particle_df is the DataFrame for the specific particle
        if particle_df.empty:
            return np.nan, np.nan
        return particle_df['x'].iloc[0], particle_df['y'].iloc[0]

    # Removed: get_peaks_data (functionality moved to analysis_utils.get_peaks_data)
    # The local self.get_peaks method is for UI peak table updates only.

    def print_ensemble_results(self, data):

        # The target here is the container box: self.box5.children[7]
        plt.figure(figsize=[7, 7])
        ax = plt.gca()
        t1_filtered = pd.concat(self.t1.loc[self.t1['particle'] == particle, :]
                                for particle in data.index.values) if len(data) > 0 else self.t1
        yfig = tp.plot_traj(t1_filtered, ax=ax) # or self.t1

        counter = 0
        for particle in data.index:
            if counter < 200:
                #get x and y position
                x_text = data.loc[particle]['x']
                y_text = data.loc[particle]['y']
                #plot ID
                if self.show_id:
                    plt.text(x_text, y_text, str(particle))
                else:
                    plt.text(x_text, y_text, str(data.loc[particle]['Change_dir']))
                counter += 1
            else:
                break

        data_fig = self.easy_print(ax)
        image1 = Image(value=data_fig)
        plt.close(ax.figure)

        # Now let's plot histograms

        a_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[14, 4])
        ax1.hist(data['Mean_vel'], bins=20)
        ax1.set_ylabel('# Particles')
        ax1.set_xlabel('Average Linear Velocity [micron/s]')
        av_value_vel = np.round(np.average(data['Mean_vel']), decimals=2) if len(data) > 0 else np.nan
        title1 = 'Mean Average Velocity: ' + str(av_value_vel) + ' [micron/s]'
        ax1.set_title(title1)

        ax2.hist(data['Change_dir_sec'], bins=20)
        ax2.set_ylabel('# Particles')
        ax2.set_xlabel('Tumbling Frequency, [1/s]')
        # if self.single_trajectories:
        av_value_change = np.round(np.average(data['Change_dir_sec']), decimals=2) if len(data) > 0 else np.nan
        # else:
        #     av_value_change = np.round(np.sum(data['Change_dir'])/np.sum(data['Duration']), decimals=2) \
        #         if len(data) > 0 else np.nan
        title2 = 'Mean tumbling frequency (single traj.): ' + str(av_value_change) + ' 1/s'
        ax2.set_title(title2)
        data_fig2 = self.easy_print(ax1)
        image2 = Image(value=data_fig2)
        plt.close(ax1.figure)

        self.box5.children[11].children = [Box([image1]), HTML(), Box([image2])]
        self.box5.children[11].image1 = image1

        results_string = 'The number of trajectories analyzed was: ' + str(len(data)) + '. Click on <b>' \
                         'Report</b> to export data to an Excel file'

        self.box5.children[12].value = results_string
        self.box5.children[15].disabled = False

    def generate_report(self, b):
        # take variables: self.excel_unfiltered and self.excel_filtered and print them to excel.
        sheet1 = self.excel_unfiltered.loc[:, 'Mean_vel':'First_Time']
        sheet2 = self.excel_filtered.loc[:, 'Mean_vel':'First_Time']

        writer = pd.ExcelWriter('output.xlsx')
        sheet1.to_excel(writer, 'Unfiltered Trajectories')
        sheet2.to_excel(writer, 'Filtered Trajectories')
        writer.save()

    def update_results(self, b):

        results = b['owner'].results
        if hasattr(b['owner'], 'is_show_id_checkbox'):
            self.show_id = not self.show_id
        percentage = self.box5.children[6].children[0].value
        range = self.box5.frame_ranges.value
        filtered_results = results[results['Mean_vel'] > self.box5.lin_vel_threshold.value]
        filtered_results = filtered_results[filtered_results['Change_dir'] <= self.box5.children[6].children[1].value]
        filtered_results = filtered_results[filtered_results['firstFrame'] <= range[1]]
        filtered_results = filtered_results[filtered_results['firstFrame'] >= range[0]]
        filtered_results = filtered_results[filtered_results['Displacement'] >=
                                            filtered_results['Displacement'].max() * percentage / 100]

        self.print_ensemble_results(filtered_results)
        self.excel_filtered = filtered_results
        self.box5.children[10].results = filtered_results # updated
        self.update_adaptation_ensemble(b)
        self.calculate_tumbling_frequency_time_frame(b)

    def calculate_adaptation_time(self, b):

        time_interval_s = self.box6.children[7].value
        time_interval_frames = time_interval_s*self.frames_second.value

        # construct a dataFrame to save results.
        columns = ['Frequency',     # 0
                   'min',           # 1
                   'max']           # 2

        # create empty data frame that will contain all results.
        index_array = np.arange(time_interval_s,
                                len(self.frames)/self.frames_second.value + time_interval_s,
                                time_interval_s)
        index_array = np.round(index_array, 3)
        results = pd.DataFrame(np.nan, index=index_array, columns=columns)
        results.index.name = 'Time'

        # The variable time_interval_frames represent the number of frames that need to be analyzed. Now we need to
        # construct a for loop that split the total number of frames into time_interval_frames
        for y in range(1, int(math.ceil(len(self.frames) / time_interval_frames) + 1)):

            # identify frames that belong to each group.
            initial = (y - 1) * time_interval_frames + 1
            final = y * time_interval_frames
            # call function calculate_ensemble_block using initial and final as input parameters. Output of this
            # function should be frequency of change of direction and error, calculated as min and max.
            index = np.round(final/self.frames_second.value, 3)
            results.loc[index]['Frequency'], \
            results.loc[index]['min'], \
            results.loc[index]['max'] = self.calculate_ensemble_block(int(initial), int(final))

        # print(results)
        # 3. plot results. --> Results are a scatter plot of frequency of change of direction vs. time interval
        ImageContainer = HBox()
        ImageContainer.children = [self.do_adaptation_plot(results)]
        self.box6.children[9].children = [ImageContainer]
        # create legend.
        self.box6.children[10].value = 'Showing adaptation plot calculated using a time interval of ' \
                                       + str(time_interval_s) + ' seconds. The parameters used for these calculations ' \
                                       'were: diameter: ' + str(self.diameter.value) + '; min value for mass: ' \
                                       + str(self.min_mass.value) + '; invert colors?: ' + str(self.invert.value) + \
                                       '; maximum displacement: ' + str(self.max_displacement.value) + \
                                       '; minimum number of frames: ' + str(self.number_frames.value) + '.'

        display(self.interface)

    def calculate_ensemble_block(self, initial, final):

            self.lock1 = False

            # self.box6.children = [legend8,            # 0
            #                       legend9,            # 1
            #                       frames_average,     # 2
            #                       smooth_cycles,      # 3
            #                       legend10,           # 4
            #                       lin_vel_threshold2, # 5
            #                       acc_threshold,      # 6
            #                       time_interval,      # 7
            #                       b_calculate2,       # 8
            #                       results2,           # 9
            #                       results_string2,    # 10
            #                       ]

            # create dataFrame
            columns = ['Velocity',          # 0
                       'Acceleration',      # 1
                       'Mean_vel',          # 2
                       'Change_dir',        # 3
                       'Duration',          # 4
                       'Change_dir_sec',    # 5
                       'x',                 # 6
                       'y']                 # 7

            # generate dataFrame t1
            f = tp.batch(self.frames[initial-1:final-1],
                         self.diameter.value,
                         minmass=self.min_mass.value,
                         invert=self.invert.value,
                         engine='numba',
                         processes='auto')
            duration = len(self.frames[initial-1:final-1])/self.frames_second.value
            # print("the number of frames being analyzed in each group is: " + str(len(self.frames[initial-1:final-1])))
            # print("The slice is: " )
            # print(self.frames[initial-1:final-1])
            # print("initial is: " + str(initial))
            # print("final is: " + str(final))
            # print("the duration is: " + str(duration))
            # time.sleep(60)
            t = tp.link_df(f, self.max_displacement.value,
                                memory=self.memory.value)  # maximum displacement in pixels.
            t1 = tp.filter_stubs(t, self.number_frames.value)  # the variable self.number_frames is defined in the window "Trajectories"
            # Get indices for unique particles
            y = t1.set_index(['frame', 'particle'])['y'].unstack()
            id_particles = y.columns.values

            # create empty data frame that will contain all results.
            results = pd.DataFrame(np.nan, index=id_particles, columns=columns)
            results.index.name = 'Particle'

            # create dictionaries to store vectors for velocity, smoothed velocity and acceleration.
            vel_dic = {}
            smooth_vel_dic = {}
            smooth_acc_dic = {}

            for particle in results.index.values:

                # calculate velocity
                vel, acc = self.calculate_vel(t1, particle)
                vel_dic.update({particle: vel})

                # smooth velocity vector
                smooth_cyles = self.box6.children[3].value
                n_frames = self.box6.children[2].value
                if smooth_cyles != 0 and n_frames != 0:
                    av_vel = self.calculate_average_vel(vel, threshold=n_frames)
                    for x in range(0, smooth_cyles - 1):
                        av_vel = self.calculate_average_vel(av_vel, threshold=n_frames)
                else:
                    av_vel = vel
                smooth_vel_dic.update({particle: av_vel})

                # calculate acceleration
                av_acc = self.calculate_av_acc(av_vel)
                smooth_acc_dic.update({particle: av_acc})

                # calculate average linear vel for filtering purposes
                results.loc[particle]['Mean_vel'] = np.average(av_vel.values)

                # get x and y coordinates.
                results.loc[particle]['x'], results.loc[particle]['y'] = self.get_x_y(t1, particle)

            # Get data for the current particle_id
            particle_data_full = t1[t1['particle'] == particle_id] 
            if particle_data_full.empty: continue

            particle_data_indexed = particle_data_full.set_index('frame')

            vel_df, _ = analysis_utils.calculate_vel(
                particle_data_indexed, self.frames_second.value, self.pixels_micron.value
            )
            if vel_df.empty: continue
            
            smooth_cycles_param = self.box6.children[3].value 
            n_frames_param_smooth = self.box6.children[2].value
            
            av_vel_df = vel_df.copy()
            if smooth_cycles_param > 0 and n_frames_param_smooth > 0:
                for _ in range(smooth_cycles_param):
                    av_vel_df = analysis_utils.calculate_average_vel(av_vel_df, threshold=n_frames_param_smooth)
            
            if av_vel_df.empty: continue

            av_acc_df = analysis_utils.calculate_av_acc(av_vel_df, self.frames_second.value)
            if av_acc_df.empty: continue

            if not av_vel_df.empty and len(av_vel_df.columns) > 0:
                 results.loc[particle_id]['Mean_vel'] = np.average(av_vel_df[av_vel_df.columns[0]].dropna().values)
            else:
                 results.loc[particle_id]['Mean_vel'] = np.nan

            results.loc[particle_id]['x'], results.loc[particle_id]['y'] = self.get_x_y(particle_data_full, particle_id)
            
            if not av_acc_df.empty and len(av_acc_df.columns) > 0:
                abs_av_acc_series = abs(av_acc_df[av_acc_df.columns[0]])
                # Use analysis_utils.get_peaks_data for consistency, passing the block's duration
                chng_dir, _, chng_dir_sec = analysis_utils.get_peaks_data(
                    particle_data_full, # For duration calculation based on this particle's presence in block
                    self.frames_second.value,
                    abs_av_acc_series, 
                    self.box6.children[6].value # acc_threshold from adaptation UI for this block
                )
                # The duration returned by analysis_utils.get_peaks_data is overall duration of particle.
                # For block analysis, we need to use the duration_per_block.
                results.loc[particle_id]['Change_dir'] = chng_dir
                results.loc[particle_id]['Duration'] = duration_per_block # Key: use fixed block duration
                results.loc[particle_id]['Change_dir_sec'] = chng_dir / duration_per_block if duration_per_block > 0 else 0
            else:
                results.loc[particle_id]['Change_dir'] = 0
                results.loc[particle_id]['Duration'] = duration_per_block
                results.loc[particle_id]['Change_dir_sec'] = 0

            # At this point, the data frame 'results' contain data for all particles. Now we need to filter for minimum
            # linear velocity threshold.
            # self.box5.children[4].results = results
            filtered_results = results[results['Mean_vel'] > self.box6.children[5].value]

            # now the next step is to print results to trajectories figure and histograms.
            data = filtered_results
            av_value_change = np.round(np.average(data['Change_dir_sec']), decimals=2) \
                if data['Change_dir_sec'].count() > 0 else np.nan # len(data)
            min_value = av_value_change - np.nanstd(data['Change_dir_sec']) / math.sqrt(data['Change_dir_sec'].count())\
                if data['Change_dir_sec'].count() > 0 else np.nan
            max_value = av_value_change + np.nanstd(data['Change_dir_sec']) / math.sqrt(data['Change_dir_sec'].count())\
                if data['Change_dir_sec'].count() > 0 else np.nan

            return av_value_change, min_value, max_value
            # All information neccesary for the calculation of av_change of direction is
            # contained in the data structure data['Change_dir_sec']

    def get_peaks_data_block(self, acc_series_abs, duration_of_block, mph=None):
        # This local helper can be simplified or removed if analysis_utils.get_peaks_data
        # is adapted or used carefully for block-based duration.
        # For now, let's assume it's a local interpretation for block processing.
        
        if acc_series_abs.empty or acc_series_abs.isna().all():
            return 0.0, duration_of_block, 0.0

        data_np = acc_series_abs.dropna().values
        peakind = detect_peaks(data_np, mph=mph) 
        number_peaks = len(peakind)

        if number_peaks == 1: 
            number_peaks = 2

        change_dir = np.floor(number_peaks / 2)
        
        frequency = 0.0
        if duration_of_block > 0:
            frequency = change_dir / duration_of_block
        elif change_dir > 0: 
            frequency = np.inf

        return change_dir, duration_of_block, frequency

    def do_adaptation_plot(self, data, threshold=0):

        plt.figure(figsize=[7, 7])
        ax = plt.gca()
        ax.plot(data['Frequency'], color='blue', marker='o', linewidth=1.5, markersize=8);
        ax.plot(data['min'], color='grey', linewidth=0.75, linestyle='dashed');
        ax.plot(data['max'], color='grey', linewidth=0.75, linestyle='dashed');

        if self.adaptation_curve_smooth_data is not None:
            data = self.adaptation_curve_smooth_data
            ax.plot(data['Frequency'], color='grey', marker='o', linewidth=1.5, markersize=8);

        if threshold != 0:
            ax.plot([data.index.values[0], data.index.values[-1]], [threshold, threshold],
                    linewidth=1, color='black')
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Tumbling Frequency, [1/s]')
        ax.set_xlabel('Time Intervals')

        if threshold != 0:
            adaptation_time, max_frequency, min_frequency = self.get_adaptation_value(data, threshold)
            title = 'The adaptation time is: ' + str(adaptation_time) + ' seconds.' + \
                    'Chng. Dir. [min; max] is: ' + '[' + str(min_frequency) + '; ' + str(max_frequency) + ']' +\
                    '\n Try a threshold value for Chn. Dir. (1/s) of: ' + str(min_frequency + (max_frequency - min_frequency)/2)
            ax.set_title(title)

        data_fig = self.easy_print(ax)
        plt.close(ax.figure)
        image1 = Image(value=data_fig)
        return image1

    def get_adaptation_value(self, data, threshold):

        data = data.dropna(how='all')
        adaptation_time = np.nan

        if (data[data['Frequency'] < threshold]['Frequency'].count() > 0) and \
           (data[data['Frequency'] > threshold]['Frequency'].count() > 0):

            try:
                    for w in data['Frequency'].values:
                        if w > threshold:
                            f2 = w
                            t2 = data[data['Frequency'] == f2].index.values[0]
                            t1 = t2 - self.box5.children[8].value
                            f1 = data['Frequency'][t1]
                            adaptation_time = (t2 - t1) / (f2 - f1) * (threshold - f2) + t2
                            break
            except KeyError: # pragma: no cover
                print("Adaptation time calculation error: Key not found. Try another threshold or check time intervals.")

        else:
            adaptation_time = np.nan

        return np.round(adaptation_time, 2), np.round(data['Frequency'].max(), 2), np.round(data['Frequency'].min(), 2)

    def get_single_trajectory(self, ti):
        # This function returns an image widget with a single trajectory.
        plt.figure(figsize=[6, 6])
        ax = plt.gca()
        yfig = tp.plot_traj(ti, ax=ax)  # or self.t1
        data_fig = self.easy_print(ax)
        image1 = Image(value=data_fig)
        plt.close(ax.figure)
        return image1

    def filter_initial_trajectories(self, b):

        # this function should update the figure contained in the field self.box_trajectories_ensemble2.

        percentage = self.box3.controller_displ.value
        frame_range = self.box3.controller_time_frame.value

        self.calculate_3_point_displacement(frame_range)

        # get % threshold contained in b and generate data DataFrame using the new column displacement.
        data = self.displacement_3_points[self.displacement_3_points['Displacement'] >=
                                          self.max_displacement_3_points * percentage/100]
        data = data[data['firstFrame'] >= frame_range[0]]
        data = data[data['firstFrame'] <= frame_range[1]]


        plt.figure(figsize=[7, 7])
        ax = plt.gca()
        t1_filtered = pd.concat(self.t1.loc[self.t1['particle'] == particle, :]
                                for particle in data.index.values) if len(data) > 0 else self.t1
        yfig = tp.plot_traj(t1_filtered, ax=ax) if len(t1_filtered) > 0 else tp.plot_traj(self.t1, ax=ax)

        x = t1_filtered.set_index(['frame', 'particle'])['x'].unstack()
        y = t1_filtered.set_index(['frame', 'particle'])['y'].unstack()
        id_particles = x.columns.values

        ## plot particle id for the first 100 particles.
        counter = 0
        for particle in id_particles:
            if counter < 200:
                # get x and y position
                x_text = x[np.isfinite(x[particle])][particle].iloc[0]
                y_text = y[np.isfinite(y[particle])][particle].iloc[0]
                # plot ID
                plt.text(x_text, y_text, str(particle), fontsize=10)
                counter += 1
            else:
                break

        data_fig = self.easy_print(ax)
        plt.close(ax.figure)
        self.box_trajectories_ensemble2.children = [Image(value=data_fig)]

    def calculate_3_point_displacement(self, range):
        # this function should populate self.displacement_3_points which is a dataframe and self.max_displacement_3_points.
        # refer to function calculate_ensemble.

        if self.max_displacement_3_points is None:
            # create dataFrame
            columns = ['Displacement', 'firstFrame']  # 0

            # Get indices for unique particles
            y = self.t1.set_index(['frame', 'particle'])['y'].unstack()
            id_particles = y.columns.values

            # create empty data frame that will contain all results.
            results = pd.DataFrame(np.nan, index=id_particles, columns=columns)
            results.index.name = 'Particle'

            for particle in results.index.values:
                # calculate displacement & first Frame
                results.loc[particle]['Displacement'], results.loc[particle]['firstFrame'] = get_displacement(self.t1,
                                                                                                              particle)
            self.displacement_3_points = results
        self.max_displacement_3_points = self.displacement_3_points['Displacement'].max() if not self.displacement_3_points.empty else 0
    else: # Recalculate max based on current range if points already exist
        # Ensure frame_range is correctly named 'range' in the original context if this was a direct copy.
        # Here, 'frame_range_val' is used for clarity, assuming it's passed as 'range' in original.
        current_frame_range = frame_range_val 
        results_in_range = self.displacement_3_points[
            (self.displacement_3_points['firstFrame'] >= current_frame_range[0]) &
            (self.displacement_3_points['firstFrame'] <= current_frame_range[1])
        ]
        self.max_displacement_3_points = results_in_range['Displacement'].max() if not results_in_range.empty else 0

    def populate_training_set(self, b):

        number_trajectories = int(self.box4.children[6].children[0].children[0].value)
        training_set = [HBox([Dropdown(options=self.current_ids,
                                       value=self.current_ids[random.randint(0, len(self.current_ids)-1)],
                                       description='Trajectory #'),
                              Dropdown(options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                                       value='0',
                                       description='# Chg. Dir:')
                              ])
                        for _ in range(number_trajectories)]
        self.box4.children[7].children = training_set
        self.box4.children[9].disabled = False
        self.populate_controller_box_genetic_algorithm()

    def load_training_set(self, b):

        e_file = self.box4.children[6].children[1].children[0].value
        if e_file == '':
            print("Please provide the name of the excel file")
            return
        try:
            df = pd.read_excel(e_file)
        except:
            print("Excel file does not exist in the provided directory. Please check for typos and try again. "
                  "Do not forget to include the extension .xlsx in the file name")
            return

        number_trajectories = df.shape[0]

        # update # Trajectories slider
        self.box4.children[6].children[0].children[0].value = number_trajectories
        self.populate_training_set(None)

        # Loop over self.box4.children[7].children to update the training set according to the data in df.
        training_set = self.box4.children[7].children #

        tr_id = list(df.columns)[0]
        tumbles = list(df.columns)[1]

        for index, row in df.iterrows():
            try:
                training_set[index].children[0].value = row[tr_id]
                training_set[index].children[1].value = str(row[tumbles])
            except:
                print("Trajectory ID = {} with an observed number of tumbles of {} is not contained in the set of "
                      "observed trajectories. Please adjust the training set correspondingly".format(str(row[tr_id]),
                                                                                                     str(row[tumbles])))

    def populate_controller_box_genetic_algorithm(self):
        advanced_opt_button = Button(description='Show Advanced Options')
        advanced_opt_button.on_click(self.show_hide_advanced_options)

        controllers_box = VBox()
        self.box4.children[8].children = [advanced_opt_button, controllers_box]

        controllers = VBox()
        label1 = HTML(value='Set population size and number of generations:')
        generations = widgets.BoundedIntText(value=5, min=1, max=10000, step=1,
                                             description='# Gens',
                                             disabled=False,
                                             continuous_update=False)
        population = widgets.BoundedIntText(value=100, min=1, max=3000, step=1,
                                            description='Ppltion S.',
                                            disabled=False, continuous_update=False)
        label2 = HTML(value='Set weights for objective functions:')
        weights_ob1 = widgets.BoundedFloatText(value=1, min=0, max=10, step=0.05,
                                               description='Weight Obj1',
                                               disabled=False,
                                               continuous_update=False)
        weights_ob2 = widgets.BoundedFloatText(value=0, min=0, max=10, step=0.05,
                                               description='Weight Obj2',
                                               disabled=False,
                                               continuous_update=False)
        label3 = HTML(value='Set upper boundaries for generation of individuals:')
        frames = widgets.BoundedIntText(value=5, min=0, max=100, step=1, description='# Frames',
                                        disabled=False, continuous_update=False)
        smooth = widgets.BoundedIntText(value=5, min=0, max=100, step=1, description='# Smooth',
                                        disabled=False, continuous_update=False)
        acceleration = widgets.BoundedIntText(value=200, min=0, max=10000, step=1, description='Acc. Thrshold',
                                              disabled=False, continuous_update=False)
        parallel = Checkbox(value=True, description='Compute in Parallel')

        self.generations = generations
        self.population = population
        self.parallel = parallel
        self.weights_obj1 = weights_ob1
        self.weights_obj2 = weights_ob2
        self.individuals_bounds = [frames, smooth, acceleration]

        controllers.children = [label1,
                                population, generations,
                                #label2,
                                #weights_ob1, #weights_ob2,
                                label3,
                                frames, smooth, acceleration,
                                parallel]

        advanced_opt_button.controllers_box = controllers

    def show_hide_advanced_options(self, b):
        if b.description == 'Show Advanced Options':
            b.description = 'Hide Advanced Options'
            self.box4.children[8].children[1].children = [b.controllers_box]
        else:
            b.description = 'Show Advanced Options'
            self.box4.children[8].children[1].children = []

    def prepare_genetic_algorithm(self, b):

        #transform data from widget into dictionary or dataFrame
        widget = self.box4.children[7]
        id_particles = [int(a.children[0].value) for a in widget.children]
        chng_dir = [int(a.children[1].value) for a in widget.children]
        columns = ['real_chng_dir', 'estimated_chng_dir', 'delta_sqr']
        training_data = pd.DataFrame(np.nan, index=id_particles, columns=columns)
        training_data.index.name = 'Particle'
        training_data['real_chng_dir'] = chng_dir

        # check for duplicates.
        training_data = training_data[~training_data.index.duplicated(keep='first')]
        self.training_data = training_data

        # call genetic algorithm routine
        b.disabled = True
        b.description = 'Calculating ...'
        self.register_deap(training_data)
        b.description = 'Estimate Parameters'
        b.disabled = False

    def eval_fitness_function(self, training_data_df, bounds, individual_ga_params):
        """
        DEAP evaluation function wrapper. Calls the core evaluation logic from analysis_utils.
        This method is registered with DEAP toolbox.
        """
        # self.t1, self.frames_second.value, self.pixels_micron.value are instance attributes.
        # bounds come from self.individuals_bounds[i].value.
        
        # The full_eval_fitness_function in analysis_utils handles iterating through training_data_df
        # and calling core_eval_fitness for each particle.
        total_error, processing_time = analysis_utils.full_eval_fitness_function(
            self.t1, 
            training_data_df, 
            self.frames_second.value, 
            self.pixels_micron.value,
            bounds, 
            individual_ga_params # These are the raw params from DEAP (e.g., 1-10)
        )
        
        # DEAP expects a tuple of fitness values.
        # The original code used weights for two objectives: (-obj1, -obj2).
        # If obj2 (e.g., time) is used, it should be returned here.
        # Assuming FitnessMin = base.Fitness, weights=(-self.weights_obj1.value, -self.weights_obj2.value)
        # If self.weights_obj2.value is 0, then processing_time won't affect fitness.
        # For now, let's match the structure if two weights are possible.
        # If only total_error is the objective, this should be (total_error,).
        # The original eval_fitness_function returned (total_error, toc-tic)
        # And creator.FitnessMin had two weights. So, we should return two values.
        return total_error, processing_time


    def register_deap(self, training_data_df): # Parameter renamed for clarity

        generations = self.generations.value
        population = self.population.value
        parallel = self.parallel.value
        bounds = [self.individuals_bounds[0].value,
                  self.individuals_bounds[1].value,
                  self.individuals_bounds[2].value]

        if parallel is False:
                # Individual definition.
                IND_SIZE = 3
                if 'FitnessMin' not in dir(creator):
                    creator.create("FitnessMin", base.Fitness, weights=(-self.weights_obj1.value,
                                                                        -self.weights_obj2.value))
                else:
                    del creator.FitnessMin
                    creator.create("FitnessMin", base.Fitness, weights=(-self.weights_obj1.value,
                                                                        -self.weights_obj2.value))

                if 'Individual' not in dir(creator):
                    creator.create("Individual", list, fitness=creator.FitnessMin)
                else:
                    del creator.Individual
                    creator.create("Individual", list, fitness=creator.FitnessMin)

                toolbox = base.Toolbox()
                toolbox.register("attr_float", random.random)
                toolbox.register("attr_int", random.randint, 1, 10) 
                toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=IND_SIZE)
                toolbox.register("evaluate", self.eval_fitness_function, training_data_df, bounds) 
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("mutate", tools.mutGaussian, mu=5.0, sigma=2.0, indpb=0.2) # Adjusted mu, sigma for 1-10 range
                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("select", tools.selNSGA2)
                pop = toolbox.population(n=population)
                halloffame = tools.HallOfFame(2)
                algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=True, halloffame=halloffame)
                
                best_params_transformed = analysis_utils.transform_parameters(halloffame[0], bounds)
                best1_serial = list(best_params_transformed)

        else: # Parallel execution
                generate_deap_excel_data(self.t1, training_data_df, self.frames_second.value, 
                                         self.pixels_micron.value, self.weights_obj1.value, 
                                         self.weights_obj2.value, bounds)
                # Call external process
                p = platform.platform()[:5]
                if p == 'Linux':
                    call_string = 'python -m scoop /opt/conda/lib/python3.6/site-packages/taxispy/run_deap_scoop.py generations='
                else:
                    call_string = 'python -m scoop /anaconda2/envs/DesignSpaceNewestPython/lib/python3.6/site-packages/' \
                                  'taxispy/run_deap_scoop.py generations='
                call_string += str(generations)
                call_string += ' population='
                call_string += str(population) 
                
                try:
                    # Ensure command is safe if generations/population could be non-numeric
                    # However, they come from BoundedIntText, so should be fine.
                    # Using shell=True is generally risky if any part of call_string is from less trusted input.
                    # For fixed paths and integer parameters, it's less of an immediate risk here.
                    # A list-based command with subprocess.run would be safer:
                    # cmd_list = ['python', '-m', 'scoop', script_path, f'generations={generations}', f'population={population}']
                    # subprocess.run(cmd_list, check=True)
                    subprocess.run(call_string, shell=True, check=True) 
                except subprocess.CalledProcessError as e:
                    print(f"Error during parallel DEAP execution: {e}")
                    return 
                except FileNotFoundError:
                    print(f"Error: Scoop or Python not found for parallel execution.")
                    return

                try:
                    results_df = pd.read_excel('results_deap.xlsx', sheet_name='results_deap', index_col=0)
                    frames_av_res = results_df.loc['frames_av']['Value']
                    smooth_res = results_df.loc['smooth']['Value']
                    acc_thresh_res = results_df.loc['acceleration_threshold']['Value']
                    best1_parallel_transformed = [frames_av_res, smooth_res, acc_thresh_res]
                    
                    halloffame_raw_df = pd.read_excel('results_deap.xlsx', sheet_name='halloffame', index_col=0)
                    # Assuming halloffame from external script is raw (1-10 like) params
                    # And index might be string "0", "1", "2" or int 0, 1, 2
                    # Ensure robust index access:
                    raw_param_indices = halloffame_raw_df.index.astype(str) if pd.api.types.is_numeric_dtype(halloffame_raw_df.index) else halloffame_raw_df.index
                                        
                    halloffame_parallel_raw = [halloffame_raw_df.loc[idx]["Value"] for idx in sorted(raw_param_indices)[:IND_SIZE]]


                except FileNotFoundError:
                    print("Error: results_deap.xlsx not found after parallel execution.")
                    return
                except KeyError as e:
                    print(f"Error reading results_deap.xlsx, sheet or column missing: {e}")
                    return

        if parallel:
            self.optimal_parameter_set = best1_parallel_transformed # Already transformed
            raw_params_for_obj_recalc = halloffame_parallel_raw
        else: 
            self.optimal_parameter_set = best1_serial # Already transformed
            raw_params_for_obj_recalc = halloffame[0] # halloffame[0] from serial is raw

        # Recalculate objective function value for the best raw parameters to store it.
        # The eval_fitness_function (wrapper) returns a tuple, so take the first element for error.
        # It's important that self.optimal_objective_function stores the actual fitness tuple from DEAP.
        self.optimal_objective_function = self.eval_fitness_function(training_data_df, bounds, raw_params_for_obj_recalc)

        self.update_optimal_parameter_set(frames_av=self.optimal_parameter_set[0],
                                          smooth=self.optimal_parameter_set[1],
                                          acceleration_threshold=self.optimal_parameter_set[2])
        self.best_parameter_set_html()

    def best_parameter_set_html(self):
        print_button = Button(description='Show Parameters')
        print_button.on_click(self.print_best_parameter_set)
        self.box4.children[10].children = [print_button, Box()]

    def print_best_parameter_set(self, b):

        html_widget_box = self.box4.children[10].children[1]
        html_widget = HTML()
        html_widget_box.children = [html_widget]

        parameter_names = ['# Frames', '# Smooth', 'Acc. Thrhld']

        if b.description == 'Show Parameters':
            b.description = 'Hide Parameters'

            s = '<div><table>\n<caption>Optimal Parameter Values: </caption>\n'
            s += '<tr align=center>' \
                 '<td style="padding:0 15px 0 15px;"><b>{0}</b></td>' \
                 '<td style="padding:0 15px 0 15px;"><b>{1}</b></td>'.format(
                '   Parameter   ',
                '  Value  ')
            s += '</tr>'
            for i in range(3):
                s += '<tr align=center>' \
                     '<td style="padding:0 15px 0 15px;">{0}</td>' \
                     '<td style="padding:0 15px 0 15px;">{1}</td>'.format(
                      parameter_names[i],
                      self.optimal_parameter_set[i],)
            s += '<table><caption>'
            s += 'Obj1: ' + str(self.optimal_objective_function[0]) + '.'
            s += '</caption></div>'
            html_widget.value = s

        else:
            html_widget.value = ''
            b.description = 'Show Parameters'

    def update_optimal_parameter_set(self, frames_av=4, smooth=3, acceleration_threshold=10):

        # update Parameter Determination Window
        self.individual_controllers_box.children[2].value = frames_av
        self.individual_controllers_box.children[3].value = smooth
        self.individual_controllers_box.children[4].value = acceleration_threshold

        # update Adaptation Times Method.
        self.box5.children[1].value = frames_av
        self.box5.children[2].value = smooth
        self.box5.acceleration.value = acceleration_threshold


def get_displacement(t, particle):

    t_i = t[t['particle'] == particle]
    first_frame = t_i['frame'].iloc[0]
    length = len(t_i.index)

    x1 = t_i['x'].iloc[0]
    y1 = t_i['y'].iloc[0]

    x2 = t_i['x'].iloc[int(length/2)]
    y2 = t_i['y'].iloc[int(length/2)]

    x3 = t_i['x'].iloc[length-1]
    y3 = t_i['y'].iloc[length-1]

    displacement = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) + \
                   math.sqrt((x2 - x3)*(x2 - x3) + (y2 - y3)*(y2 - y3))

    return displacement, first_frame


# Removed: transform_parameters (now in analysis_utils)

def generate_deap_excel_data(t1_df, training_data_df, frames_second_val, pixels_micron_val, 
                             obj1_val, obj2_val, bounds_list):
    """ Prepares an Excel file with data needed for the external run_deap_scoop.py script. """
    # Ensure consistent naming with how analysis_utils.py might receive them if it were reading this.
    # For run_deap_scoop.py, it reads specific sheet names.
    config_data = {
        'frames_second': frames_second_val,
        'pixels_micron': pixels_micron_val,
        'obj1_weight': obj1_val, # Use more descriptive names for clarity
        'obj2_weight': obj2_val,
        'Frames_bound': bounds_list[0],
        'Smooth_bound': bounds_list[1],
        'Acceleration_bound': bounds_list[2]
    }
    config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])

    try:
        with pd.ExcelWriter('deap_excel_data.xlsx') as writer:
            t1_df.to_excel(writer, sheet_name='t1_trajectories') 
            training_data_df.to_excel(writer, sheet_name='training_data_details') # Use more descriptive sheet names
            config_df.to_excel(writer, sheet_name='config_and_bounds', index=False)
    except Exception as e: 
        print(f"Error writing DEAP Excel data for external script: {e}")
        # This error should be handled, e.g., by not proceeding with parallel execution.

def generate_adaptation_string(data, smooth=False):

    sheet1 = t1
    sheet2 = training_data
    index = ['frames_second', 'pixels_micron', 'obj1', 'obj2', 'Frames', 'Smooth', 'Acceleration']
    columns = ['Value']
    values = [frames_second, pixels_micron, obj1, obj2, bounds[0], bounds[1], bounds[2]]
    sheet3 = pd.DataFrame(values, index=index, columns=columns)
    sheet3.index.name = 'Parameter'

    writer = pd.ExcelWriter('deap_excel_data.xlsx')
    sheet1.to_excel(writer, 't1')
    sheet2.to_excel(writer, 'training_data')
    sheet3.to_excel(writer, 'video_properties_constraints')
    writer.save()


def generate_adaptation_string(data, smooth=False):

    if smooth is True:
        s = '<div><table>\n<caption>Showing Data for Smoothed Adaptation Curve: </caption>\n'
    else:
        s = '<div><table>\n<caption>Showing Data for Raw Adaptation Curve: </caption>\n'
    s += '<tr align=center>' \
         '<td style="padding:0 15px 0 15px;"><b>{0}</b></td>' \
         '<td style="padding:0 15px 0 15px;"><b>{1}</b></td>' \
         '<td style="padding:0 15px 0 15px;"><b>{2}</b></td>'.format(
         '  Time [s]  ',
         '  Tumbling Frequency [1/s]  ',
         '  # Trajectories  ')
    s += '</tr>'
    for i in data.index:
        s += '<tr align=center>' \
             '<td style="padding:0 15px 0 15px;">{0}</td>' \
             '<td style="padding:0 15px 0 15px;">{1}</td>' \
             '<td style="padding:0 15px 0 15px;">{2}</td>'.format(
              i,
              data.loc[i]['Frequency'],
              data.loc[i]['Number'])

    return s
