# CS470_EmotionDetection
CS470 Final Project, emotion detection with masked data

** This project is based on the 'Deep-Emotion (https://github.com/omarsayed7/Deep-Emotion)' paper **

## 1. Architecture
## 2. Datasets
    JAFFE
    CK+
    FER2013 skdjfl;jalgkj;

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
    
Make dataset/jaffedbase, dataset/CK+48 and fer2013.csv file.   
Make a new folder which name is 'dataset' and insert all things inside it.   
Make sure that dataset/jaffedbase, dataset/CK+48, dataset/fer2013.csv.  

To make 'train_jaffe_Mask_원본.csv', 'train_jaffe_NoMask_원본.csv', 'masked_jaffedbase' folder that consists of masked images.  
    
    $python gen_masked.py [-data] [--dataset] J

To make 'train_ckplus_Mask_원본.csv', 'train_ckplus_NoMask_원본.csv', 'mask_CKplus' folder that consists of masked images
    
    $python gen_masked.py [-data] [--dataset] C

To make 'train_fer2013_Mask_원본.csv', 'train_fer2013_NoMask_원본.csv'   
RUN this three times 
    
    $python gen_masked.py [-data] [--dataset] F

by uncommenting the below parts one by one at sequence in 'gen_masked.py' file   

    data = data[:11962]
    # data = data[11962:23924]
    # data = data[23924:]

    df.to_csv('./dataset/train_fer2013_NoMask_원본_frac1.csv')
    # df.to_csv('./dataset/train_fer2013_NoMask_원본_frac2.csv')
    # df.to_csv('./dataset/train_fer2013_NoMask_원본_frac3.csv')

    df.to_csv('./dataset/train_fer2013_Mask_원본_frac1.csv')
    # df.to_csv('./dataset/train_fer2013_Mask_원본_frac2.csv')
    # df.to_csv('./dataset/train_fer2013_Mask_원본_frac3.csv')

Finally, combine three files into one, and name it 'train_fer2013_Mask_원본.csv' (or 'train_fer2013_NoMask_원본.csv')  

###    b) How to run
&nbsp;&nbsp;&nbsp;&nbsp;Setup the datset

            python main.py [-s [True]] [-d [data_path]]
                --setup                     Setup the datset for the first time
                --data                       Data folder that contains data files
                
&nbsp;&nbsp;&nbsp;&nbsp;To train the model

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
                                                                
        
    

