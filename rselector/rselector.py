#!/home/mtx/workspace/anaconda3/bin/python3

import argparse
# from RsModel import RModel
import os
from configparser import ConfigParser

def convert_var(var):
    if var=='None':
        var=None
    return var


def make_config(i):
    with open("default.conf",'w') as fd:
        fd.write(
            '''
[Training]
    Train_directory=
    Valid_directory=None
    Validation_ratio=0.3
    PreWeights=None
    SavePath=default
    
    GPU=-1
    Train_epochs=7
    Steps_per_epoch=600  
    
[NetworkParameters]
    F_alpha=1
    Backbone_network=vgg19
    Max_detection=400
    
[Predict]
    Model_weights=
    Images_directory=
    Boxes_outputdir=
    Threshold=
    
    GPU=0
    NMS_threshold=0.6
            '''
        )

def train(args):
    config_file = args.config
    cp = ConfigParser()
    cp.read(config_file)

    train_section = cp.sections()[0]
    train_dir = cp.get(train_section,'train_directory')
    valid_dir = cp.get(train_section,'valid_directory')
    valid_dir = convert_var(valid_dir)
    valid_ratio = cp.get(train_section,'validation_ratio')
    valid_ratio = float(valid_ratio)
    preweights = cp.get(train_section,'preweights')
    preweights = convert_var(preweights)
    savepath = cp.get(train_section,'savepath')
    savepath = convert_var(savepath)
    gpu = cp.get(train_section,'gpu')
    gpu = int(gpu)
    epochs = cp.get(train_section,'train_epochs')
    epochs = int(epochs)
    steps = cp.get(train_section,'steps_per_epoch')
    steps = int(steps)

    param_section = cp.sections()[1]
    alpha = cp.get(param_section,'f_alpha')
    backbone = cp.get(param_section,'backbone_network')
    # max_det = cp.get(param_section,'max_detection')
    params = {
        'alpha':int(alpha),
        'backbone':backbone,
        # 'max_det':int(max_det)
    }

    from RsModel import RModel
    rmodel = RModel()
    rmodel.train(train_dir,valid_dir,valid_ratio,gpu=gpu,weights=preweights,save_path=savepath,epochs=epochs,steps=steps,param=params)


def predict(args):
    config_file = args.config
    cp = ConfigParser()
    cp.read(config_file)

    predict_section = cp.sections()[2]
    weights = cp.get(predict_section,'model_weights')
    image_dir = cp.get(predict_section,'images_directory')
    output = cp.get(predict_section,'boxes_outputdir')
    threshold = cp.get(predict_section,'threshold')
    gpu = cp.get(predict_section,'gpu')
    nms_threshold = cp.get(predict_section,'nms_threshold')

    param_section = cp.sections()[1]
    alpha = cp.get(param_section,'f_alpha')
    backbone = cp.get(param_section,'backbone_network')
    # max_det = cp.get(param_section,'max_detection')
    params = {
        'alpha':int(alpha),
        'backbone':backbone,
        # 'max_det':int(max_det)
    }

    from RsModel import RModel
    rmodel = RModel()
    rmodel.set_session(gpu)
    rmodel.load(weights)
    rmodel.predict(image_dir,threshold,nms=True,output=output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(help="rs method")

    parser_train = subparser.add_parser('train',description="training the model")
    parser_train.add_argument("config",type=str,help="config file of parameters in training,you can use 'rselector make-config' to create default config file")
    parser_train.set_defaults(function=train)

    parser_defaultconfig = subparser.add_parser('make-default')
    parser_defaultconfig.set_defaults(function=make_config)

    parser_predict = subparser.add_parser('predict',description="predict boxfile of micrographs")
    parser_predict.add_argument("config",type=str,help="config file in predicting,you can use 'rselector make-config' to create default config file")
    parser_predict.set_defaults(function=predict)

    args = parser.parse_args()
    args.function(args)