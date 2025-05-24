from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import os
import subprocess # Added import
import ipywidgets as widgets
from IPython.display import display # Required if running in Jupyter to display widgets

# Importing specific widgets for clarity
VBox = widgets.VBox
Text = widgets.Text
Button = widgets.Button
HTML = widgets.HTML # For displaying messages


class Splitter(object):
    """
    A class providing a simple UI to split video files into frames using ffmpeg.
    """

    def __init__(self):
        """
        Initializes the Splitter UI components.
        Sets up input fields for source video and output directory,
        a button to trigger splitting, and an area for messages.
        """
        self.path_in = Text(description='Source Video:',
                            placeholder='/path/to/your/video.mov',
                            style={'description_width': 'initial'},
                            layout=widgets.Layout(width='95%'))
        self.path_out = Text(description='Output Directory:',
                             placeholder='/path/to/output/frames_folder',
                             style={'description_width': 'initial'},
                             layout=widgets.Layout(width='95%'))

        self.b_split = Button(description='Split Video',
                              tooltip='Click to start splitting the video into frames')
        self.b_split.on_click(self.split_video)

        self.output_message = HTML(value="") # For displaying status and error messages

        self.interface = VBox(children=[self.path_in, 
                                         self.path_out, 
                                         self.b_split, 
                                         self.output_message],
                              layout=widgets.Layout(padding='10px'))

    def split_video(self, b):
        """
        Handles the video splitting process when the 'Split Video' button is clicked.

        Retrieves input/output paths, creates the output directory if needed,
        and executes ffmpeg via subprocess.run() to extract frames.
        Provides user feedback on success or failure.

        Args:
            b: The button widget instance that triggered this callback.
        """
        self.b_split.disabled = True
        self.b_split.description = 'Splitting...'
        self.output_message.value = "<p>Processing...</p>"

        input_path = self.path_in.value
        output_path = self.path_out.value

        if not input_path or not output_path:
            self.output_message.value = "<p style='color:red;'>Error: Both source video and output directory paths must be specified.</p>"
            self.b_split.description = 'Split Video'
            self.b_split.disabled = False
            return

        try:
            os.makedirs(output_path, exist_ok=True)
        except OSError as e:
            self.output_message.value = f"<p style='color:red;'>Error creating output directory '{output_path}': {e}</p>"
            self.b_split.description = 'Split Video'
            self.b_split.disabled = False
            return
        except Exception as e:
            self.output_message.value = f"<p style='color:red;'>An unexpected error occurred while creating directory: {e}</p>"
            self.b_split.description = 'Split Video'
            self.b_split.disabled = False
            return

        output_pattern = os.path.join(output_path, 'output-%05d.jpg')
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-y', # Overwrite output files without asking
            output_pattern
        ]

        try:
            process = subprocess.run(ffmpeg_cmd, 
                                     capture_output=True, 
                                     text=True, 
                                     check=False)

            if process.returncode == 0:
                if any(fname.lower().startswith('output-') and fname.lower().endswith('.jpg') for fname in os.listdir(output_path)):
                    self.output_message.value = f"<p style='color:green;'>Success! Video split into frames in '{output_path}'.</p>"
                else:
                    self.output_message.value = (
                        f"<p style='color:orange;'>ffmpeg ran successfully, but no output frames (output-*.jpg) "
                        f"were found in '{output_path}'.</p>"
                        f"<details><summary>ffmpeg output (click to expand)</summary>"
                        f"<p><b>Command:</b><br/><code>{' '.join(ffmpeg_cmd)}</code></p>"
                        f"<p><b>Stdout:</b></p><pre>{process.stdout if process.stdout else 'N/A'}</pre>"
                        f"<p><b>Stderr:</b></p><pre>{process.stderr if process.stderr else 'N/A'}</pre>"
                        f"</details>"
                    )
            else:
                error_details = (
                    f"<p style='color:red;'>ffmpeg execution failed with return code {process.returncode}.</p>"
                    f"<details><summary>Error details (click to expand)</summary>"
                    f"<p><b>Command:</b><br/><code>{' '.join(ffmpeg_cmd)}</code></p>"
                    f"<p><b>Stderr:</b></p><pre>{process.stderr if process.stderr else 'N/A'}</pre>"
                    f"<p><b>Stdout:</b></p><pre>{process.stdout if process.stdout else 'N/A'}</pre>"
                    f"</details>"
                )
                self.output_message.value = error_details

        except FileNotFoundError:
            self.output_message.value = ("<p style='color:red;'>Error: ffmpeg command not found. "
                                         "Please ensure ffmpeg is installed and included in your system's PATH.</p>")
        except Exception as e:
            self.output_message.value = f"<p style='color:red;'>An unexpected error occurred during ffmpeg execution: {e}</p>"
        
        finally:
            self.b_split.description = 'Split Video'
            self.b_split.disabled = False