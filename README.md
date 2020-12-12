# CS470_EmotionDetection
CS470 Final Project, emotion detection with masked data

** This project is based on the 'Deep-Emotion (https://github.com/omarsayed7/Deep-Emotion)' paper **

## 1. Architecture
## 2. Datasets
    JAFFE
    CK+
    FER2013 skdjfl;

## 3. Prerequisites
Install mtcnn
    
    $pip install mtcnn

## 4. Structure of this repository
main.py : setup of the dataset and training loop  
visualize.py : the source code for evaluating the model on the data  
deep_emotion.py : the model class  
data_loaders.py : the dataset class  
generate_data : setup of the dataset  
gen_masked.py : the source code for generating the masked data  

## 5. Usage
###    a) Data Preparation

Download the dataset from following links:   
&nbsp;&nbsp;&nbsp;&nbsp;JAFFE -  https://zenodo.org/record/3451524.  
&nbsp;&nbsp;&nbsp;&nbsp;CK+ - https://www.kaggle.com/shawon10/ckplus.  
&nbsp;&nbsp;&nbsp;&nbsp;FER2013 - https://www.kaggle.com/deadskull7/fer2013.  
    
Make data/jaffedbase, data/CK+48 and data/fer2013.csv file.   
Make a new folder which name is 'data' and insert all things inside it.   
Make sure that data/jaffedbase, data/CK+48, data/fer2013.csv.  

To make 6 files and 1 folder as follows   
'train_jaffe_Mask_원본.csv'   
'train_jaffe_Mask.csv'   
'test_jaffe_Mask.csv'   
'train_jaffe_NoMask_원본.csv'   
'train_jaffe_NoMask.csv'   
'test_jaffe_NoMask.csv'   
'masked_jaffedbase' folder that consists of masked images.  
    
    $python gen_masked.py -data J

To make 6 files and 1 folder   
'train_ckplus_Mask_원본.csv'   
'train_ckplus_Mask.csv'   
'test_ckplus_Mask.csv'   
'train_ckplus_NoMask_원본.csv'   
'train_ckplus_NoMask.csv'   
'test_ckplus_NoMask.csv'   
'mask_CKplus' folder that consists of masked images.  
    
    $python gen_masked.py -data C

To make 6 files   
'train_fer2013_Mask_원본.csv'   
'train_fer2013_Mask.csv'   
'test_fer2013_Mask.csv'   
'train_fer2013_NoMask_원본.csv'   
'train_fer2013_NoMask.csv'   
'test_fer2013_NoMask.csv'   
    
    $python gen_masked.py -data F


###    b) How to run
#### &nbsp;&nbsp;&nbsp;&nbsp;Setup the datset
Change the target file names into 'train.csv', and 'test.csv', respectively.   
   
For example, if you want to run the code with no masked Jaffe dataset, then change 'train_jaffe_NoMask.csv' and 'test_jaffe_NoMask.csv' into 'train.csv', and 'test.csv', respectively.  
   
Then run below code.   

            python main.py [-s [True]] [-d [data_path]]
                --setup                     Setup the datset for the first time
                --data                       Data folder that contains data files
                
                ## Run the following code:
                    python main.py -s True -d './data'
                
#### &nbsp;&nbsp;&nbsp;&nbsp;To train the model

Make sure that the target files(train.csv, test.csv, val.csv, finaltest.csv, train folder, val folder, finaltest folder) that you want to use for training are the files from the target dataset.   

            python main.py [-t] [--data [data_path]] [--hparams [hyperparams]]
                                        [--epochs] [--learning_rate] [--batch_size]
                                        [--channel50] [--stn] [--regulizer]
                --data                       Data folder that contains training and validation files
                --train                      True when training
                --hparams                    True when changing the hyperparameters
                --epochs                     Number of epochs
                --learning_rate              Learning rate value
                --batch_size                 Training/validation batch size
                --channel50                  To modify the channel number as 50, set True
                --stn                        To remove stn process, set False
                --regulizer                  To apply regulization, set True
                
                ## Run the following code:
                    Original deep_emotion model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 
                    No STN model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --stn False
                    Channel_50 model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --channel50 True
                    L1_0.001 model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --regulizer True

#### &nbsp;&nbsp;&nbsp;&nbsp;To validate the model

Make sure that the data path(finaltest folder) should contain the test images for the target dataset.   

            python visualize.py [-t] [-c] [--data  [data_path]] [--file  [file_path]] [--model [model_path]]
                                        [--channel50] [--stn] [--regulizer]
            
                --data                       Path to the finaltest folder that contains finaltest images
                --file                       Path to the finaltest.csv
                --model                      Path to pretrained model
                --test_cc                    Returns test accuracy and visualization of confusion matrix
                --saliency_map               Returns saliency map for 10 test images
                --channel50                  If the pretrained model's channel number is 50, set True
                --stn                        If the pretrained model ignored stn process, set False
                --regulizer                  If the pretrained model used regulization, set True
                
                ## Run the following code:
                    Original pretrained deep_emotion model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' ## Enter the model path
                        
                    No STN pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --stn False
                        
                    Channel_50 pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --channel50 True

                    L1_0.001 pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --regulizer True

                    
        
    

