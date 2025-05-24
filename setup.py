import setuptools

# It's good practice to ensure README.md exists, or handle its absence
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A Python tool for analyzing particle trajectories from image sequences."

# Dependencies from environment.yml, excluding python itself
# and pip-only if any were there.
install_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "ipywidgets",
    "pims",
    "trackpy",
    "deap",
    "scoop",
    "ipython", # For IPython.display
    "openpyxl", # For pandas Excel I/O
]

setuptools.setup(
    name="taxispy",
    version="0.1.6.3", # From InitGui.py
    author="Miguel A. Valderrama-Gomez", # From InitGui.py
    author_email="mvalderramag@example.com", # Placeholder
    description="A Python tool for analyzing particle trajectories from image sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m1vg/taxispy", # Assumed repository URL
    license="MIT", # From InitGui.py
    packages=setuptools.find_packages(where="."), # Explicitly state to look in current directory
    package_dir={"": "."}, # Specify that packages are under the root directory
    install_requires=install_requires,
    python_requires=">=3.8", # Based on environment.yml
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta", # Or "3 - Alpha"
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # Consider adding an entry point if the GUI can be launched from a function
    # For example:
    # entry_points={
    #     'console_scripts': [
    #         'taxispygui=taxispy.InitGui:launch_gui_function', # Requires a launch_gui_function in InitGui
    #     ],
    # },
)
