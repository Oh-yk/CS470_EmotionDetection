from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import os
import cv2
import scipy as sp
import numpy as np
import mtcnn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pandas import Series, DataFrame
import torch

np.random.seed(470)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Jaffe():
    data_path = './data/jaffedbase'
    data_dir_list = os.listdir(data_path)
    dict_data = dict(emotion= [], pixels= [])
    
    for dataset in data_dir_list:
        input_img=plt.imread(data_path+'/'+ dataset)
        if (input_img.shape == (256, 256, 4)):
            input_img = input_img[:, :, 0]
        assert(input_img.shape == (256, 256))
        input_img = input_img.reshape((1, 256*256))

        pixels_str = ''
        for i in input_img[0]:
            pixels_str += f'{i}' + ' '
        pixels_str = pixels_str[:-1]

        if 'AN' in dataset:
            label = 0
        elif 'DI' in dataset:
            label = 1
        elif 'FE' in dataset:
            label = 2
        elif 'HA' in dataset:
            label = 3
        elif 'NE' in dataset:
            label = 6
        elif 'SA' in dataset:
            label = 4
        elif 'SU' in dataset:
            label = 5
        # 합쳐서 dataframe에 저장
        dict_data['emotion'].append(label)
        dict_data['pixels'].append(pixels_str)
    # dict to dataframe
    df = DataFrame(dict_data)
    # dataframe to csv
    df.to_csv('./data/train_jaffe_NoMask_원본.csv')
    print('Making csv file for nomask jaffedbase dataset finish')
    
    if not os.path.exists('./data/masked_jaffedbase'):
        os.mkdir('./data/masked_jaffedbase')
    data_path = './data/jaffedbase'
    data_dir_list = os.listdir(data_path)
    
    for dataset in data_dir_list:
        input_img=plt.imread(data_path+'/'+ dataset)
        if ('KM' in dataset) or ('KA.AN1.39.tiff' == dataset):
            input_img = input_img[:, :, :-1]
        else:
            input_img = np.expand_dims(input_img, axis=2)
            temp_img=np.append(input_img, input_img, axis=2)
            input_img=np.append(temp_img, input_img, axis=2)
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(input_img)
        pixels = input_img.copy()
        for result in faces:
            x, y, width, height = result['box']
            eye_y = result['keypoints']['right_eye'][1]
            nose_y = result['keypoints']['nose'][1]
            eps = 0.2;
            bound_y = eye_y*eps+nose_y*(1-eps)
            bound_height = y+height-bound_y
            pixels[int(bound_y):int(bound_y+bound_height), x:x+width, :] = 255
        save_paths = f'./data/masked_jaffedbase/{dataset}'
        plt.imsave(save_paths, pixels)
    print('Making masked images for Jaffe finish')
    
    data_path = './data/masked_jaffedbase'
    data_dir_list = os.listdir(data_path)
    dict_data = dict(emotion= [], pixels= [])
    
    for dataset in data_dir_list:
        input_img=plt.imread(data_path+'/'+ dataset)
        if (input_img.shape == (256, 256, 4)):
            input_img = input_img[:, :, 0]
        input_img = input_img.reshape((1, 256*256))

        pixels_str = ''
        for i in input_img[0]:
            pixels_str += f'{i}' + ' '
        pixels_str = pixels_str[:-1]

        if 'AN' in dataset:
            label = 0
        elif 'DI' in dataset:
            label = 1
        elif 'FE' in dataset:
            label = 2
        elif 'HA' in dataset:
            label = 3
        elif 'NE' in dataset:
            label = 6
        elif 'SA' in dataset:
            label = 4
        elif 'SU' in dataset:
            label = 5
        # 합쳐서 dataframe에 저장
        dict_data['emotion'].append(label)
        dict_data['pixels'].append(pixels_str)
    # dict to dataframe
    df = DataFrame(dict_data)
    # dataframe to csv
    df.to_csv('./data/train_jaffe_Mask_원본.csv')
    print('Making csv file for masked jaffedbase dataset finish')
    
    data = pd.read_csv('./data/train_jaffe_NoMask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_jaffe_NoMask.csv')
    test.to_csv('./data/test_jaffe_NoMask.csv')
    
    data = pd.read_csv('./data/train_jaffe_Mask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_jaffe_Mask.csv')
    test.to_csv('./data/test_jaffe_Mask.csv')
    print('Finish splitting into train and test set')
    
def Fer():
    data = pd.read_csv('./data/fer2013.csv')
    for k in range(3):
        if (k==0):
            raw_data = data[:11962]
        elif (k==1):
            raw_data = data[11962:23924]
        elif (k==2):
            raw_data = data[23924:]
        raw_data = raw_data.sample(frac=0.1, replace=True, random_state=1)

        dict_data = dict(emotion= [], pixels= [])
        dict_mask_data = dict(emotion= [], pixels= [])
        
        for index in tqdm(range(raw_data.shape[0])):
            label = raw_data['emotion'].iloc[index]
            img_arr_str = raw_data['pixels'].iloc[index].split(' ')
            img_arr = np.asarray(img_arr_str, dtype=np.uint8).reshape(48, 48)
            input_img = Image.fromarray(img_arr)
            # resize into (48, 48, 3)
            input_img = np.expand_dims(input_img, axis=2)
            temp_img=np.append(input_img, input_img, axis=2)
            input_img=np.append(temp_img, input_img, axis=2)
            detector = mtcnn.MTCNN()
            faces = detector.detect_faces(input_img)
            if faces != []:
                pixels = input_img.copy()
                for result in faces:
                    x, y, width, height = result['box']
                    eye_y = result['keypoints']['right_eye'][1]
                    nose_y = result['keypoints']['nose'][1]
                    eps = 0.2;
                    bound_y = eye_y*eps+nose_y*(1-eps)
                    bound_height = y+height-bound_y
                    pixels[int(bound_y):int(bound_y+bound_height), x:x+width, :] = 255
                if (pixels.shape == (48, 48, 3)):
                    pixels = pixels[:, :, 0]
    
                assert(pixels.shape == (48, 48))
                pixels = pixels.reshape((1, 48*48))
    
                pixels_str = ''
                for i in pixels[0]:
                    pixels_str += f'{i}' + ' '
                pixels_str = pixels_str[:-1]
                # no mask 용
                input_img = input_img[:, :, 0]
                assert(input_img.shape == (48, 48))
                input_img = input_img.reshape((1, 48*48))
                nomask_pixels = ''
                for i in input_img[0]:
                    nomask_pixels += f'{i}' + ' '
                nomask_pixels = nomask_pixels[:-1]
                dict_mask_data['emotion'].append(label)
                dict_mask_data['pixels'].append(pixels_str)
                dict_data['emotion'].append(label)
                dict_data['pixels'].append(nomask_pixels)
        df = DataFrame(dict_data)
        df.to_csv(f'./data/train_fer2013_NoMask_원본_frac{k}.csv')

        df = DataFrame(dict_mask_data)
        df.to_csv(f'./data/train_fer2013_Mask_원본_frac{k}.csv')
    
    f1_path = './data/train_fer2013_NoMask_원본_frac0.csv'
    f1_mask_path = './data/train_fer2013_Mask_원본_frac0.csv'
    
    f2_path = './data/train_fer2013_NoMask_원본_frac1.csv'
    f2_mask_path = './data/train_fer2013_Mask_원본_frac1.csv'
    
    f3_path = './data/train_fer2013_NoMask_원본_frac2.csv'
    f3_mask_path = './data/train_fer2013_Mask_원본_frac2.csv'
    
    alldata = []
    df1 = pd.read_csv(f1_path)
    alldata.append(df1)
    df2 = pd.read_csv(f2_path)
    alldata.append(df2)
    df3 = pd.read_csv(f3_path)
    alldata.append(df3)
    
    dataCombine=pd.concat(alldata, axis=0, ignore_index=True)
    dataCombine.to_csv('./data/train_fer2013_NoMask_원본.csv')
    
    allmaskdata = []
    mask_df1 = pd.read_csv(f1_mask_path)
    allmaskdata.append(mask_df1)
    mask_df2 = pd.read_csv(f2_mask_path)
    allmaskdata.append(mask_df2)
    mask_df3 = pd.read_csv(f3_mask_path)
    allmaskdata.append(mask_df3)
    
    datamaskCombine=pd.concat(allmaskdata, axis=0, ignore_index=True)
    datamaskCombine.to_csv('./data/train_fer2013_Mask_원본.csv')
    
    data = pd.read_csv('./data/train_fer2013_NoMask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_fer2013_NoMask.csv')
    test.to_csv('./data/test_fer2013_NoMask.csv')
    
    data = pd.read_csv('./data/train_fer2013_Mask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_fer2013_Mask.csv')
    test.to_csv('./data/test_fer2013_Mask.csv')
    print('Finish splitting into train and test set')
    
def CKplus():

    data_path = './data/CK+48'
    data_dir_list = os.listdir(data_path)
    
    if not os.path.exists('./data/CKplus'):
        os.mkdir('./data/CKplus')
        os.mkdir('./data/CKplus/anger')
        os.mkdir('./data/CKplus/contempt')
        os.mkdir('./data/CKplus/disgust')
        os.mkdir('./data/CKplus/fear')
        os.mkdir('./data/CKplus/happy')
        os.mkdir('./data/CKplus/sadness')
        os.mkdir('./data/CKplus/surprise')
        
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+dataset)
        for img in img_list:
            im1 = Image.open(f'./data/CK+48/{dataset}/{img}')
            string = img.replace('png', 'jpg')
            im1.save(f'./data/CKplus/{dataset}/{string}')
            
    data_path = './data/CKplus'
    data_dir_list = os.listdir(data_path)
    dict_data = dict(emotion= [], pixels= [])
    
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+dataset)
        for img in img_list:
            input_img=plt.imread(data_path+'/'+ dataset+'/'+img)
            assert(input_img.shape == (48, 48))
            input_img = input_img.reshape((1, 48*48))

            pixels_str = ''
            for i in input_img[0]:
                pixels_str += f'{i}' + ' '
            pixels_str = pixels_str[:-1]

            if 'anger' == dataset:
                label = 0
            elif 'disgust' == dataset:
                label = 1
            elif 'fear' == dataset:
                label = 2
            elif 'happy' == dataset:
                label = 3
            elif 'contempt' == dataset:
                label = 6
            elif 'sadness' == dataset:
                label = 4
            elif 'surprise' == dataset:
                label = 5
            # 합쳐서 dataframe에 저장
            dict_data['emotion'].append(label)
            dict_data['pixels'].append(pixels_str)
    # dict to dataframe
    df = DataFrame(dict_data)
    # print(df.head())
    # dataframe to csv
    df.to_csv('./data/train_ckplus_NoMask_원본.csv')
    print('Making NoMask dataset csv file finish')
    
    data_path = './data/CKplus'
    data_dir_list = os.listdir(data_path)
    if not os.path.exists('./data/mask_CKplus'):
        os.mkdir('./data/mask_CKplus')
        os.mkdir('./data/mask_CKplus/anger')
        os.mkdir('./data/mask_CKplus/contempt')
        os.mkdir('./data/mask_CKplus/disgust')
        os.mkdir('./data/mask_CKplus/fear')
        os.mkdir('./data/mask_CKplus/happy')
        os.mkdir('./data/mask_CKplus/sadness')
        os.mkdir('./data/mask_CKplus/surprise')
        
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=plt.imread(data_path + '/'+ dataset + '/'+ img )
            input_img = np.expand_dims(input_img, axis=2)
            temp_img=np.append(input_img, input_img, axis=2)
            input_img=np.append(temp_img, input_img, axis=2)
            detector = mtcnn.MTCNN()
            faces = detector.detect_faces(input_img)
            pixels = input_img.copy()
            for result in faces:
                x, y, width, height = result['box']
                eye_y = result['keypoints']['right_eye'][1]
                nose_y = result['keypoints']['nose'][1]
                eps = 0.2;
                bound_y = eye_y*eps+nose_y*(1-eps)
                bound_height = y+height-bound_y
                pixels[int(bound_y):int(bound_y+bound_height), x:x+width, :] = 255
            save_paths = f'./data/mask_CKplus/{dataset}/{img}'
            plt.imsave(save_paths, pixels)
    print('Making masked images in mask_CKplus folder finish')
    
    data_path = './data/mask_CKplus'
    data_dir_list = os.listdir(data_path)
    dict_mask_data = dict(emotion= [], pixels= [])
    
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+dataset)
        for img in img_list:
            input_img=plt.imread(data_path+'/'+ dataset+'/'+img)
            input_img = input_img[:, :, 0]
            assert(input_img.shape == (48, 48))
            input_img = input_img.reshape((1, 48*48))

            pixels_str = ''
            for i in input_img[0]:
                pixels_str += f'{i}' + ' '
            pixels_str = pixels_str[:-1]

            if 'anger' == dataset:
                label = 0
            elif 'disgust' == dataset:
                label = 1
            elif 'fear' == dataset:
                label = 2
            elif 'happy' == dataset:
                label = 3
            elif 'contempt' == dataset:
                label = 6
            elif 'sadness' == dataset:
                label = 4
            elif 'surprise' == dataset:
                label = 5
            # 합쳐서 dataframe에 저장
            dict_mask_data['emotion'].append(label)
            dict_mask_data['pixels'].append(pixels_str)
    # dict to dataframe
    mask_df = DataFrame(dict_mask_data)
    # dataframe to csv
    mask_df.to_csv('./data/train_ckplus_Mask_원본.csv')
    print('Making Mask dataset csv file finish')
    
    data = pd.read_csv('./data/train_ckplus_NoMask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_ckplus_NoMask.csv')
    test.to_csv('./data/test_ckplus_NoMask.csv')
    
    data = pd.read_csv('./data/train_ckplus_Mask_원본.csv')
    train, test = train_test_split(data, random_state=2)
    train.to_csv('./data/train_ckplus_Mask.csv')
    test.to_csv('./data/test_ckplus_Mask.csv')
    print('Finish splitting into train and test set')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generation of masked dataset")
    parser.add_argument('-data', '--dataset', type=str, help='J for Jaffedbase, F for FER2013, C for CKplus')

    args = parser.parse_args()
    
    if args.dataset:
        dataset = args.dataset
    else:
        dataset = 'J'

    if (dataset is 'J'):
        Jaffe()
    elif (dataset is 'F'):
        Fer()
    elif (dataset is 'C'):
        CKplus()
