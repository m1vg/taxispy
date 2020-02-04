from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import os
from IPython.display import display
import ipywidgets as widgets
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


class Splitter(object):

    def __init__(self):
        self.path_in = Text(description='Source',
                          placeholder='/Documents/videoA.mov ..')
        self.path_out = Text(description='Output',
                           placeholder='/Documents/videoA ..')

        b_split = Button(description='Split')
        b_split.on_click(self.split_video)

        self.interface = VBox(children=[self.path_in, self.path_out, b_split])

    def split_video(self, b):
        b.disabled = True
        b.description = 'Splitting..'
        # if folder does not exist, create it.
        if not os.path.exists(self.path_out.value):
            os.system('mkdir ' + self.path_out.value)
            #os.makedirs(self.path_out.value)

        # split!
        s = 'ffmpeg -i '
        s += self.path_in.value
        s += ' ' + self.path_out.value + '/output-%05d.jpg'
        os.system(s)
        b.description = 'Split'
        b.disabled = False

        #check if target directory is empty
        if len(os.listdir(self.path_out.value)) == 0:
            print('Splitter failed. Check paths and try again!')
        else:
            print('Success!')