# cdti_voxel
Repoistory for Code related to "The Effect of Voxel-Volume and Voxel-Shape on Cardiac Diffusion Tensor Imaging Metrics" (in prep)

# Overview
Code to create figures is in the "Code" folder while Data will be available in the "Data" Folder. Withing the Code folder there are 3 mains folders: **01_Processing**, **02_Figures**, and **Software**. The Processing folder contains code to process the Dicom images that came off of the scanner. Original dicoms are not provided, but NifTis will be available. Each processing step is saved into a seperate folder. In the **02_Figures** folder, each jupyter notebook corresponds to generating figures and analysis for Mean Diffusivity (MD), Fractional Anisotropy (FA), Helix Angle Pitch (HAP), and uncertainty of MD (dMD), FA (dFA), and primary eigenvector (dE1).
The final ranked assessment of all metrics together is in the final notebook (**fig_TOPSIS.ipynb**). Each jupyter notebook is formatted as "fig_{Metric}.ipynb".

For processing data and accessing colormaps, the CarDpy toolbox (https://github.com/tecork/CarDpy) is utilized and is located inthe Software folder to mitigate potential dependencies issues. 

# Setup 
Jupyter notebooks were creating using Python 3.8.18 with Anacodona. An enviornment file has been provided to help configure the python enviornment and install packages. 

