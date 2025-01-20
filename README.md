Master Thesis - Jimmy Billen
-
# Master's Thesis: Hidden Information Behind the Time Series: Extracting nullcline structures using artificial neural networks

This repository contains the code and results from my master's thesis, conducted under the supervision of Prof. Dr. L. Gelens, Dr. N. Frolov, and MSc. B. Prokop at the Laboratory of Dynamics in Biological Systems.

## Thesis Overview

In this thesis, we investigate the use of artificial neural networks to recover the nullclines of biological oscillating systems. The primary model studied is the FitzHugh-Nagumo model, a well-known analytically solved biological oscillator.

### Main Investigations:
1. **Hyperparameter Optimization**: Focused on minimizing both the validation error and the nullcline prediction error to enhance the model's accuracy.
2. **Algorithm Robustness**: Tested the robustness of the neural network under variations in the time-scale parameter and in scenarios with limited data availability.
3. **Generalization to Complex Systems**: Applied the algorithm to a more complex system with two bicubic nullclines, demonstrating its capacity to generalize beyond simple models.
4. **Preliminary Investigation of Novak and Tyson Oscillator**: Explored the potential of the neural network approach on a more intricate biological oscillator model.

## Repository Contents
- **Code**: Data generation of the biological oscillators, implementation of the neural network and analysis of the hyperparameter optimization process.
- **Data**: Training data of the neural networks.
- **Results**: Analysis of the model’s performance and generalization capabilities.

For more detailed information, the full thesis can be accessed in the repository or via the institution's repository.

## Contact
For questions, feel free to contact me directly.

# SETUP
Decompile Files:
-
First, decompile the following files:
- FHN_NN_loss_and_model_7.5_15000.zip
- FHN_NN_loss_and_model_100_15000.zip

Install Dependencies
-
Install the required Python packages using the provided requirements.txt file. Run the following command in your terminal:
pip install -r requirements.txt


/project_root

├── settings.py                  # Main configuration file for system parameters

├── /data_generation_exploration # Scripts for generating and exploring the data

├── /model_building              # Scripts related to training and saving neural networks

├── /data_analysis               # Scripts for analyzing, visualizing, and interpreting data

├── /data                        # Output data (CSV files, saved models)

└── /extras                      # Additional scripts or experimental work

**For more insight into every program.**
The most important programs are explained in the file: overview_programs.txt

For creation of new neural networks for new settings and new data use following workflow:
-
Tune the right parameters in settings.py
Configure the time series in data_generation_exploration/FitzHugh_Nagumo*
Create a dataframe where the data of the NN can be saved in model_building/create_dataframe.py
Train the models with the desired configuration in model_building/create_NN_FHN.py
Perform the required data analysis using data_analysis/
- Validation Error: loss_function_plot.py / loss_function_thesis_plot.ipynb
- Nullcline Error: Nullcline_MSE_plot_thesis.py (NN selection => Nullcline Error Calculation + Saving => Nullcline Error Analysis)
- Neural Network Configuration Analysis: NN_model_analysis
