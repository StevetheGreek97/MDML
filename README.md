# Project title: Molecular_Dynamics_&_Machine_Learning

This repository is for the graduation projrct for the Master in Data Science 
for Life Sciences at Hanze University of Applied Sciences.

The purpose of this project would be to explore new methods for overcoming molecular dynamics problems by using ideas from machine learning (ML). 
The objective is to explore ML/AI models to predict the effect of mutations on EGFR as obtained from much shorter simulations,
cutting down both time and produced data. Questions that would be answer are: 
    1. Is it possible to predict long term simulations from sort term ones? 
    2. How short is short enough? 
    
Student:          Stylianos Mavrianos, s.mavrianos@st.hanze.nl 
Supervisor:       Tsjerk Wassenaar, t.a.wassenaar@pl.hanze.nl 
Daily supervisor: Vilacha Madeira R Santos, j.f.vilacha@rug.nl 

## Requirements

- Python 3.8.10
- Numpy
- MatplotLib
- PLotly
- Scikit-learn
- Tensorflow
- Keras
- MDAnalysis
- cv2
- yaml

## Setup

1. Clone the repository to your local machine:

shell: git clone https://github.com/StevetheGreek97/MD_ML.git

2. Create a new environment:

shell: virtualenv MD_ML

3. Install the required packages:

pip install -r requirments.txt

4. Run jupyter notebook to check the tutorials in the examples folder.

## Usage

The Preprocessing.py, Machinelearing.py and Mapping.py are the three parts of the pipeline. These modules use the rest. All you have to to is to configure a yaml configuration file (conf.yml) that has a masterpath to e folder that contain subfolders of each classification state (e.g., active, inactive state). Whithin each subfoloder a .pdb and .xtc of that state should be included.
-data+
     |
     +--active---+
     |           |
     |           +- topology file (.pdb)
     |           |
     |           +- coordinates file (.xtc)
     |
     +--inactive-+
                 |
                 +- topology file (.pdb)
                 |
                 +- coordinates file (.xtc)
