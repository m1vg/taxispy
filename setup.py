import setuptools

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A Python tool for analyzing particle trajectories from image sequences."

install_requires = [
    "numpy",
    "pandas",
    "matplotlib",
    "ipywidgets",
    "pims",
    "trackpy",
    "deap",
    "scoop",
    "ipython",
    "openpyxl",
]

setuptools.setup(
    name="taxispy",
    version="0.1.6.3",
    author="Miguel A. Valderrama-Gomez",
    author_email="miguel.valderrama.gomez@gmail.com",
    description="A Python tool for analyzing cell trajectories from image sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m1vg/taxispy",
    license="MIT",
    packages=setuptools.find_packages(where="."), 
    package_dir={"": "."},
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]
)
