---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

Author: Shubham Kumar Singh
Date: May 16, 2020
Email_id: shubham.singh@mtech2017.iitgn.ac.in
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

Directory structure: (tree product_detection_shubham_singh)

product_detection_shubham_singh
│ 
├── Assignment\ Solution.pdf
├── GroceryDataset_part2	(***downloaded GroceryDataset_part2 from github***)
│   ├── BrandImagesFromShelves
│	├── ProductImagesFromShelves
├── Readme.md
├── ShelfImages	(***Copy train/test images here***)
│   ├── test
│	└── train
├── pack_detector
│   ├── Output    (***Contain test images with anchor boxes***)
│ 	├── annotation.txt
│   ├── data
│   │   ├── eval.record
│   │   ├── pack.pbtxt
│   │   └── train.record
│   ├── image2products.json
│   ├── metrics.json
│   ├── model.h5
│   ├── models   (***train/evel Auto generated from Tensorlfow***)
│   │   └── ssd_mobilenet_v1
│   │       ├── eval     
│	│		├── ssd_mobilenet_v1_pack.config
│   │       └── train
│	├── pack_detector_fg
│   │   └── frozen_inference_graph.pb
│   └── tmp.json
├── product_detection_shubham_singh.ipynb
└── requirement.txt
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------


1. Dataset: (The dataset to be used for training/testing is the Grocery dataset.)
	(i) Replaced GroceryDataset_part1 with given ShelfImages directory


2. product_detection_shubham_singh.ipynb (Source code) 
	(i) Data preparation, training and evaluation scripts are present inside this
    file.
	(ii) Single ipython file name product_detection_shubham_singh.ipynb is present
    in product_detection_shubham_singh folder which you get after extracting the 
    zip file.
	(iii) No path need to provide as code is written such a way that It w'll check
    your present working directory (pwd) and set all paths for files present inside
    uncompress folder. 

3. requirement.txt
	(i) This file contain all the required library need for this code to be run.

4. Output:
	(i) image2products.json file present inside pack_detector directory.
	(ii) metrices.json file present inside pack_detector directory.
	(iii) files in test directory saved inside output directory 
	with anchor boxes.

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Project Details

I have implemented two methods. 

---------------------------------------------------------------------------------
First is using Keras library for ResNet CNN. This is end to end architecture for 
SSD for product recognisition. Second is using Tensorflow Object Detection API.
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------


1. Data Generation: (ImageDataGenerator API)
	ImageDataGenerator attributes are as following:

	rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally 
    height_shift_range=0.1,  # randomly shift images vertically 
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

    Left-right flip augmentation is used so that an image should be equally
    recognizable as its horizontal mirror image.
    Vertical_flip augmentation is used so that an image should be equally 
    recognizable as its Virtical mirror image.

2. model.summary()

________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 120, 80, 3)   0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 120, 80, 16)  448         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 120, 80, 16)  64          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 120, 80, 16)  0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 120, 80, 16)  2320        activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 120, 80, 16)  64          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 120, 80, 16)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 120, 80, 16)  2320        activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 120, 80, 16)  64          conv2d_3[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 120, 80, 16)  0           activation_1[0][0]               
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 120, 80, 16)  0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 120, 80, 16)  2320        activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 120, 80, 16)  64          conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 120, 80, 16)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 120, 80, 16)  2320        activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 120, 80, 16)  64          conv2d_5[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 120, 80, 16)  0           activation_3[0][0]               
                                                                 batch_normalization_5[0][0]      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 120, 80, 16)  0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 120, 80, 16)  2320        activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 120, 80, 16)  64          conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 120, 80, 16)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 120, 80, 16)  2320        activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 120, 80, 16)  64          conv2d_7[0][0]                   
__________________________________________________________________________________________________
add_3 (Add)                     (None, 120, 80, 16)  0           activation_5[0][0]               
                                                                 batch_normalization_7[0][0]      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 120, 80, 16)  0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 60, 40, 32)   4640        activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 60, 40, 32)   128         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 60, 40, 32)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 60, 40, 32)   9248        activation_8[0][0]               
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 60, 40, 32)   544         activation_7[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 60, 40, 32)   128         conv2d_9[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 60, 40, 32)   0           conv2d_10[0][0]                  
                                                                 batch_normalization_9[0][0]      
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 60, 40, 32)   0           add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 60, 40, 32)   9248        activation_9[0][0]               
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 60, 40, 32)   128         conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 60, 40, 32)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 60, 40, 32)   9248        activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 60, 40, 32)   128         conv2d_12[0][0]                  
__________________________________________________________________________________________________
add_5 (Add)                     (None, 60, 40, 32)   0           activation_9[0][0]               
                                                                 batch_normalization_11[0][0]     
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 60, 40, 32)   0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 60, 40, 32)   9248        activation_11[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 60, 40, 32)   128         conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 60, 40, 32)   0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 60, 40, 32)   9248        activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 60, 40, 32)   128         conv2d_14[0][0]                  
__________________________________________________________________________________________________
add_6 (Add)                     (None, 60, 40, 32)   0           activation_11[0][0]              
                                                                 batch_normalization_13[0][0]     
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 60, 40, 32)   0           add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 30, 20, 64)   18496       activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 30, 20, 64)   256         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 30, 20, 64)   0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 30, 20, 64)   36928       activation_14[0][0]              
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 30, 20, 64)   2112        activation_13[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 30, 20, 64)   256         conv2d_16[0][0]                  
__________________________________________________________________________________________________
add_7 (Add)                     (None, 30, 20, 64)   0           conv2d_17[0][0]                  
                                                                 batch_normalization_15[0][0]     
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 30, 20, 64)   0           add_7[0][0]                      
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 30, 20, 64)   36928       activation_15[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 30, 20, 64)   256         conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 30, 20, 64)   0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 30, 20, 64)   36928       activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 30, 20, 64)   256         conv2d_19[0][0]                  
__________________________________________________________________________________________________
add_8 (Add)                     (None, 30, 20, 64)   0           activation_15[0][0]              
                                                                 batch_normalization_17[0][0]     
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 30, 20, 64)   0           add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 30, 20, 64)   36928       activation_17[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 30, 20, 64)   256         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 30, 20, 64)   0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 30, 20, 64)   36928       activation_18[0][0]              
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 30, 20, 64)   256         conv2d_21[0][0]                  
__________________________________________________________________________________________________
add_9 (Add)                     (None, 30, 20, 64)   0           activation_17[0][0]              
                                                                 batch_normalization_19[0][0]     
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 30, 20, 64)   0           add_9[0][0]                      
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 3, 2, 64)     0           activation_19[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 384)          0           average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           3850        flatten_1[0][0]                  
==================================================================================================
Total params: 277,642
Trainable params: 276,266
Non-trainable params: 1,376
__________________________________________________________________________________________________


3. Hyper parameter tunning:
	batch_size = 50
	epochs = 15
	verbose=1, workers=4, 
    lr_schedule lr = 1e-3
    if epoch > 5:
        lr = lr * 1e-1
__________________________________________________________________________________________________
__________________________________________________________________________________________________

4. Key findings 

(i). Instead of searching for a logo everywhere on the shelf image, I extract the structure of the 
market shelves prior to applying object  recognition. 
(ii). in addition to the segmentation of the image into product and non-product, I propose a 
technique to determine self boundaries
(iii). Since the warning part is common across brands, we consider  the  first 40%  portion  
of the  image  from the  top  for  classification.
(iv). Potential applications of the proposed approach include planogram compliance control, 
inventory management and assisting visually impaired people during shopping.

__________________________________________________________________________________________________
__________________________________________________________________________________________________

5. Acknowledgements
The dataset is collected as part of a TUBITAK funded project carried out by Idea Teknoloji.