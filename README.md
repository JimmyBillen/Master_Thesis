Master Thesis - Jimmy Billen
-

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

To analyse the produced results use the files in /data_analysis

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
