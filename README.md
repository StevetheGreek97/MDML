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

```git clone https://github.com/StevetheGreek97/MD_ML.git```

2. Create a new environment:

``` virtualenv MD_ML```

3. Install the required packages:

```pip install -r requirments.txt```

4. Run jupyter notebook to check the tutorials in the examples folder.

## Usage

The pipeline consists of three modules: Preprocessing.py, Machinelearning.py and Mapping.py.

 To get started, simply configure a yaml configuration file (conf.yml) that includes:
1. the 'masterpath' to a folder containing subfolders for each classification state (e.g., active, inactive state) -> str

Each subfolder should contain a .pdb and .xtc file for the corresponding state.

```
 < data >
     |  
     |
     |--active
     |     |
     |     |--topology file (.pdb)
     |     |
     |     |--coordinates file (.xtc)
     |
     |--inactive
           |
           |--topology file (.pdb)
           |
           |--coordinates file (.xtc)
```

2. a 'savingpath' that all the results with be saved. -> str

3. 'downsampled_to' how many image should be created for each state -> int

```
downsample_to: 1659
masterpath: /path/to/data/
savepath: /path/to/save

```
The final output includes a series of down-sampled images, a prefomance img, a confusion matrix, a saliency map, a .txt file listing important residues, and a .pdb file with b-factor information showing the important residues. 

In order to run the pipeline simply run this code:
```
python3 main.py -c path/to/confg.yml
```
