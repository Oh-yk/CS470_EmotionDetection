# CS470_EmotionDetection
CS470 Final Project, emotion detection with masked data

** This project is based on the 'Deep-Emotion (https://github.com/omarsayed7/Deep-Emotion)' paper**

1. Architecture
2. Datasets
    JAFFE
    CK+
    FER2013

3. Prerequisites
4. Structure of this repository
main.py : setup of the dataset and training loop
visualize.py : the source code for evaluating the model on the data
deep_emotion.py : the model class
data_loaders.py : the dataset class
generate_data : setup of the dataset
gen_masked.py : the source code for generating the masked data

5. Usage
    a) Data Preparation
        Download the dataset from following links:
            JAFFE -  https://zenodo.org/record/3451524
            CK+ - https://www.kaggle.com/shawon10/ckplus
            FER2013 - https://www.kaggle.com/deadskull7/fer2013
        Make dataset/jaffedbase, dataset/CK+48, 
    b) How to run
        Setup the datset
            python main.py [-s [True]] [-d [data_path]]
                --setup                     Setup the datset for the first time
                --data                       Data folder that contains data files
        To train the model
            python main.py [-t] [--data [data_path]] [--hparams [hyperparams]]
                                        [--epochs] [--learning_rate] [--batch_size]
                                        [--m [channel50]] [--n [stn]] [--l [regulizer]]
                --data                      Data folder that contains training and validation files
                --train                      True when training
                --hparams               True when changing the hyperparameters
                --epochs                  Number of epochs
                --learning_rate         Learning rate value
                --batch_size            Training/validation batch size
                --m                          To modify the channel number, set True
                --n                           To remove stn process, set False
                --l                            To apply regulization, set True
                                                                
        
    

