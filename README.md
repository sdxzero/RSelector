# RSelector
###    RSelector is a CNN particle selector based on RetinaNet.
###
## 1 .Requirement
### Hardware
> GPU:Nvidia GTX 1080 or GPU memory>7GB
### Python Libs:
> + numpy
> + tensorflow
> + keras
> + cv2
> + mrcz
> + progressbar
> + zignor
> + keras_resnet
> + Cython
> + progressbar
> + scipy
### Install
> #### pip3 install rselector
## 2 .Dataset Format
### To start  training ,micrographs and annotations should be organized as follow format:
> #### DatasetDirectory/
> ####         |------- id_1.mrc
> ####         |------- id_1.box
> ####         |------- id_2.mrc
> ####         |------- id_2.box
> ####         |    ......    

## 3 .Params in Configuration File
### You could use command  to make a default config file as follows:
> #### rselector make-default
### Fill the requied params. Do not  change the params in 'NetworkParameters' if you don't understand what you do. And params in 'NetworkParameters' should keep consistent in training and  prediction.

> ####        [Training]
> ####  &emsp;&emsp;  Train_directory=                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Dataset path to training the network
> ####  &emsp;&emsp;  Valid_directory=None                &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# If valid data is 'None',rselector could split training dataset according to the valid_ratio 
> ####  &emsp;&emsp;  Validation_ratio=0.3              &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  If valid dataset is 'None',the ratio of split valid data
> ####  &emsp;&emsp;  PreWeights=None                   &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#   Path of model weights to fine-tune the model. 
> ####  &emsp;&emsp;  SavePath=default                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Path of  model weights to be saved
>  ####   
> ####   &emsp;&emsp; GPU=-1                            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# If GPU=-1,then cpu will be used .Otherwise, GPU should set to 0,1... 
>  ####  &emsp;&emsp; Train_epochs=7                    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#   Training epochs,the epoch param should not be less than size of training data.
> ####   &emsp;&emsp; Steps_per_epoch=600               &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# Steps in each epoch.
> ####    
> ####  [NetworkParameters]
> ####  &emsp;&emsp;  F_alpha=1                         &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# F_score params to balance precision and recall 
> ####  &emsp;&emsp;  Backbone_network=vgg19            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# Backbone of rselector.Optional:'vgg16','vgg19','resnet50','resnet101'
>  #### &emsp;&emsp;  Max_detection=400                 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# Max detection of rselector
> ####    
> ####  [Predict]
>  ####  &emsp;&emsp; Model_weights=                    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Path of trained model weights.
> ####   &emsp;&emsp; Images_directory=                 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Directory of micrographs to be predicted.
> ####  &emsp;&emsp;  Boxes_outputdir=                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Path of output coordinates .box files.
> ####  &emsp;&emsp;  Threshold=                        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  Threshold to filter the output bounding boxes.
> ####    
> ####  &emsp;&emsp;  GPU=0                             &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\#  As gpu params in [Training]
> ####  &emsp;&emsp;  NMS_threshold=0.6                 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;\# NMS_threshold is to adjust the max overlapped boxes in output .

## 4 .Train the network
### After fill the configuration file.You could start training as follows:
> #### rselector train *your.config.file*
#### Wait for  finish the training and remember the best threshold to predict micrographs. And ...
## 5 .Prediction
### Fill params in [Predict].Params in [NetworkParameters] should be consistent with Training stage.And start  predict:
> #### rselector predict *your.config.file*
## That's done...
