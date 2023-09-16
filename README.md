# CliffordBernoulli
Codebase used in generating data and making plots for this paper: https://arxiv.org/abs/2309.04520  
Data for the plots and data collapses can be found here: 

Installation:
Clone the repository and run "pip install -r requirements.txt" to install all the necessary packages (or manually install them from the list in requirements.txt). All of the scripts run on Python 3.10.11

Data generation:
All of the HPC*.py files are the files used to generate the data for the plots. These files can be ran using the format "python3 HPC*.py llll pppp rrrr" where llll = system size, pppp = probability to perform the Control Map, and rrrr = number of realizations.  
The system sizes, probabilites, and realizations used can be found in the BernoulliFinalPlots.ipynb file, or you can just download the complete dataset from the link above. Since the HPC*.py files output several .npy files per system size and probability, the file structure is not correct if you just run the HPC*.py files, but the dataset download preserves the file structure if you extract the zip files into the repository folder.

Once the dataset is generated or (more preferrably) downloaded, all of the data collapse scripts and the plot notebook will run off of this data.  

Data Processing:
For the data collapse files, the two collapses that do not involve Time Dynamic (TD in the file structure) data will output several values and errorbars for the fit. Each set of outputs is the two fit values (nu and p_c) followed by their respective errorbars, and the values used in the paper can be crossreferenced from this list.  

The Time Dynamic fits are much more straight forward, and these calculate z values and their errobars that can be found in Table 1 of the paper.  

Finally, there is the plot notebook that constructs the figures from the paper. This is also straight forward once all the data is downloaded/structured and will duplicate the current figures. By default, the savefig command for each plot is commented so if you want the pdf you will have to go uncomment them.  
