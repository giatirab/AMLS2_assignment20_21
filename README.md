# README

PROJECT NAME: "SENTIMENT ANALYSIS WITH TWEETS"

The project aims at solving a Kaggle competition using Natural Language Processing tools. The task is to classify a tweet's embedded emotion as either positive or negative. 

##########          CODEBASE STRUCTURE          ########## 

The repository include 2 python files. The main file is named "main.py" and performs the tasks of dataset preprocessing, training, testing and classification once launched. It leverages on the class TransformerManager which includes the relevant methods for running the full machine learning task. Here below is a more detailed description of the purpose of each file and sub-folder in the "AMLS2_assignment20_21" GitHub repository.

1) main.py: the file contains the main class "TranformerManager" which defines the whole set of steps to solve the sentiment analysis task via its built-in methods. These include preprocessing, train, test (private function) and classify functions. The class "TransformerManager" leverages other classes defined within the modules.py file. The function "main()", which is defined outside the "TransformerManager" class, is the one actually initializing the program steps (tm.preprocess(), tm.train(), tm.classify()).

2) modules.py: this python file describes how the main building blocks of the Tranformer architecture are defined and put together to create the final deep leaning network architecture. These are class objects named SelfAttention, TransformerBlock and Transformer.

3) data folder: this is the location where the raw dataset (1.6 million tweet downloadable from https://www.kaggle.com/kazanova/sentiment140) and compressed preprocessed datasets (.gz extension) are saved. The code will uncompress these 2 files (train_dataset.csv.gz and train_dataset.csv.gz) and automatically generate the unwrapped .csv version for both. The folder also contains the parameters.json file where input parameters can be manually edited before launching the program (learning rate, number of epochs, batch sizes, log_step, etc.).

4) models folder: this is the location that the trained model are saved. The user can run the same architecture with different input parameters (either via arguments declaration in terminal or directly editing the parameters.json file). The algorithm will save down models with an "encrypted" label which is references those chosen arguments. The folder also contains the vocabularies which have been generated via the Fields object (label_fields.pt and text_fields.pt).

5) colab.ipynb: this jupyter file can be loaded into Google Colab and will clone the GitHub repository live and launch the program on the Colab GPU. Tensorboard is launched as well.

6) runs folder: tensorboard will save files within this folder while the program runs. train loss and test loss & average recall will be saved for each epoch. This is a visualisation tool which will help understand the empirical results.

##########          HOW TO RUN THE PROGRAM          ##########

This NLP task is computational intensive (in particular for training when parallelisation is performed in the deep learning tensors computation). Thus, the use of a GPU for training (not for pre-processing) is advised. The model available within the "models/ca8538c49a089d19f6d40cc6178f1e0c" file has been trained on a Google Colab GPU machine and allows classification of new tweets with a high accuracy score (83%) and confidence. However, if the user is keen on running the NLP architecture (ie the Transformer) with different input parameters, he/she can either amend the parameters.json file manually in "/parameters" folder or declare inputs in the terminal directly and pass them as program "args" ("ArgumentParser" python modules is indeed imported in main.py). Automatically a new file containing the trained model is saved into the "models" folder with its unique ID name corresponding to the parameters passed for that run.

If a local/private GPU machine is not available and the user is not keen on paying for cloud GPU resource (Google, AWS, etc..), he/she can run the "colab.ipynb" file in Google Colab environment while activating the GPU machine before launching the program. The Google Colab file will clone the public GitHub repository directly and run the main.py file. If running on personal GPU please activate the "sentiment_analaysis" anaconda environment by running the code "conda env create -f environment.yml". The libraries needed for this project are already available on Google Colab.
