# README

PROJECT NAME: "SENTIMENT ANALYSIS WITH TWEETS"

- CODEBASE STRUCTURE

The repository include X python files. The main file is named "main.py" and performs the tasks of dataset preprocessing, training, testing and classification once launched. It leverages on the class TransforrmerManager which includes the relevant class methods for running the full machine learning task. Here below is a more detailed description of the purpose of each file and folder in the "AMLS2_20-21_SN17024244" repository.

1) main.py:

2) modules.py:

3) data folder: this is the location where the raw dataset (1.6 million tweet downloadable from XYZ) and compressed preprocessed datasets (gz extension) are saved. The code will uncompress these 2 files (train_dataset.csv.gz and train_dataset.csv.gz) and automatically generate the .csv version for both. The folder also contains the parameters.json file where input parameters can be edited before launching the program (learning rate, number of epochs, batch sizes, log_step, ..).

4) models folder: this is the location that the trained model are saved. The user can run the same architecture with different input parameters (either via args declaration in terminal or directly editing the parameters.json file). The algorithm will save down models with an "encrypted" label which is references those chosen args. The folder also contains the vocabularies which have been generated via the Fields object (label_fields.pt and text_fields.pt).

5) sentiment_analysis.ipynb: this jupyter file can be loaded into Google Colab and will clone the GitLab repository live and launch the program on the Colab GPU. Tensorboard is launched as well.

6) runs folder: tensorboard will save files within this folder while the program runs. train loss and test loss & average recall will be saved for each epoch. This is a visualisation tool which will help understand the empirical results.

- HOW TO RUN THE PROGRAM?
