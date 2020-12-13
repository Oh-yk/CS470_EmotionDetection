# CS470_EmotionDetection
CS470 Final Project, emotion detection with masked data

** This project is based on the 'Deep-Emotion (https://github.com/omarsayed7/Deep-Emotion)' paper **

## 1. Architecture
The basic architecture of our project is based on the deep-emotion architecture.  
We modified the model so that it can detect facial expressions while people wearing a mask.  
We made three models(NoSTN, Channels_50, l1_0.001).  
You can train these models using codes in the 'Usage' section.  

## 2. Datasets
    JAFFE
    CK+
    FER2013

## 3. Installation
Install mtcnn
    
    $pip install mtcnn
    
To run the codes, you need to have the following libraries:
* pytorch == 1.7.0+cu101
* torchvision == 0.8.1+cu101
* mtcnn == 0.1.0
* cv2 == 4.1.2
* tqdm == 4.41.1
* PIL == 7.0.0
* sklearn == 0.22.2.post1
* matplotlib == 3.2.2
* scipy == 1.4.1
* numpy == 1.18.5
* argparse == 1.1
* seaborn == 0.11.0

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
    
The 'jaffedbase' folder, the 'CK+48' folder, and the 'fer2013.csv' file will be downloaded by upper links, respectively.   
   
You can choose any dataset that you want, but we recommend 'CK+48' dataset.   
   
Make a new folder which name is 'data' and insert all things inside it.   
   
For example, the path to the dataset folder should be './data/CK+48'.   
   

To make 6 files and 1 folder as follows   
'train_jaffe_Mask_원본.csv'   
'train_jaffe_Mask.csv'   
'test_jaffe_Mask.csv'   
'train_jaffe_NoMask_원본.csv'   
'train_jaffe_NoMask.csv'   
'test_jaffe_NoMask.csv'   
'masked_jaffedbase' folder that consists of masked images,   
run the below code.   
    
    $python gen_masked.py -data J

To make 6 files and 1 folder   
'train_ckplus_Mask_원본.csv'   
'train_ckplus_Mask.csv'   
'test_ckplus_Mask.csv'   
'train_ckplus_NoMask_원본.csv'   
'train_ckplus_NoMask.csv'   
'test_ckplus_NoMask.csv'   
'mask_CKplus' folder that consists of masked images,  
run the below code.   
    
    $python gen_masked.py -data C

To make 6 files   
'train_fer2013_Mask_원본.csv'   
'train_fer2013_Mask.csv'   
'test_fer2013_Mask.csv'   
'train_fer2013_NoMask_원본.csv'   
'train_fer2013_NoMask.csv'   
'test_fer2013_NoMask.csv',   
run the below code.   
    
    $python gen_masked.py -data F


###    b) How to run
#### &nbsp;&nbsp;&nbsp;&nbsp;Setup the dataset
Change the target file names into 'train.csv', and 'test.csv', respectively.   
   
For example, if you want to run the code with no masked CK+48 dataset, then change 'train_ckplus_NoMask.csv' and 'test_ckplus_NoMask.csv' into 'train.csv' and 'test.csv', respectively.  
   
To make 'val.csv', 'finaltest.csv', 'train folder', 'val folder', and 'finaltest folder', run the below code.   
   
You will need the files and folders for the Masked CK+48 dataset,   
so I recommend to insert all the files ('train.csv', 'test.csv', 'val.csv', 'finaltest.csv', 'train folder', 'val folder', and 'finaltest folder') into a new folder which name is 'NoMasked_ckplus'.   
   
And then repeat the setup process with the 'train_ckplus_Mask.csv' and 'test_ckplus_Mask.csv'.   
   
After running the code, insert all the files ('train.csv', 'test.csv', 'val.csv', 'finaltest.csv', 'train folder', 'val folder', and 'finaltest folder') into a new folder which name is 'Masked_ckplus'.   

            python main.py [-s [True]] [-d [data_path]]
                --setup                     Setup the dataset for the first time
                --data                       Data folder that contains data files
                
                ## Run the following code:
                    python main.py -s True -d './data'
                   
             
#### &nbsp;&nbsp;&nbsp;&nbsp;To train the model
Change the file/folder names(train.csv, test.csv, val.csv, finaltest.csv, train folder, val folder, finaltest folder) of the target dataset(e.g. Masked Jaffe) into the file/folder names as 'train.csv', 'test.csv', 'val.csv', 'finaltest.csv', 'train' folder, 'val' folder, and 'finaltest' folder, respectively.   
   
If you followed my recommendation, you don't need to change the file/folder names.   
Just write down './data/NoMasked_ckplus'(or './data/Masked_ckplus') into data_path below.   
   
Then run the below code.   
   

            python main.py [-t] [--data [data_path]] [--hparams [hyperparams]]
                                        [--epochs] [--learning_rate] [--batch_size]
                                        [--channel50] [--stn] [--regularizer]
                --data                       Data folder that contains training and validation files
                --train                      True when training
                --hparams                    True when changing the hyperparameters
                --epochs                     Number of epochs
                --learning_rate              Learning rate value
                --batch_size                 Training/validation batch size
                --channel50                  To modify the channel number as 50, set True
                --stn                        To remove stn process, set False
                --regularizer                To apply regulization, set True
                
                ## Run the following code:
                    Original deep_emotion model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 
                    No STN model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --stn False
                    Channel_50 model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --channel50 True
                    L1_0.001 model:
                        python main.py -t TRAIN --data './data' --hyperparams True --epochs 150 --learning_rate 0.004 --batch_size 64 --regularizer True

#### &nbsp;&nbsp;&nbsp;&nbsp;To test / visualize the model

Make sure that the 'data path' should indicate the finaltest folder, but its name can be changed, just write down the path to the folder that contains the test images for the target dataset.   
   
Make sure that 'file_path' is the path to the finaltest.csv made at the Setup process, but its name can be changed, just write down the path to the file.   
   
Caution: If you use the Jaffe dataset, then you should set True for the --jaffeset parameter.   
   

            python visualize.py [-t] [-c] [--data  [data_path]] [--file  [file_path]] [--model [model_path]]
                                        [--channel50] [--stn] [--regularizer] [--jaffeset]
            
                --data                       Path to the finaltest folder that contains finaltest images
                --file                       Path to the finaltest.csv
                --model                      Path to pretrained model
                --test_cc                    Returns test accuracy and visualization of confusion matrix
                --saliency_map               Returns saliency map for 10 test images
                --channel50                  If the pretrained model's channel number is 50, set True
                --stn                        If the pretrained model ignored stn process, set False
                --regularizer                If the pretrained model used regulization, set True
                --jaffeset                   If you are using the jaffe dataset, set True
                
                ## Run the following code:
                    Original pretrained deep_emotion model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' ## Enter the model path
                        
                    No STN pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --stn False
                        
                    Channel_50 pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --channel50 True

                    L1_0.001 pretrained model:
                        python visualize.py -t --data './data/finaltest' --file './data/finaltest.csv' --model 'model_path' --regularizer True

                    
        
    

