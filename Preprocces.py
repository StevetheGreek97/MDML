from Createfolders import CreateFolders as cf
from Yamlhandler import YamlHandler as yamlh
from TrajectoryTools import LoadTrajectories, ParseTrajectory, down_sample
from ImageTrasformation import xyz2rgb
import tensorflow as tf
import numpy as np


def preprocess(conf_file):
    confkeys = yamlh(conf_file) 
    output_folder = f"{confkeys.read_key('savepath')}/output"
    cf(output_folder).create_subfolders()
     
    path = confkeys.read_key('masterpath')

    print('Loading trajecrories...')
    data  = LoadTrajectories().traj_from_dir(path + 'data/')

    for (name,  trajectory) in data.items():
        traj_obj = ParseTrajectory(name, trajectory.select_atoms('name CA'))
        print(traj_obj)
        downsampled =  down_sample(traj_obj.allign(), confkeys.read_key('downsample_to'))

        confkeys.update_yaml('height', downsampled[0].shape[0])
        confkeys.update_yaml('lenght', downsampled[0].shape[1])

        downsample_rgb = xyz2rgb(downsampled)
        cf(f'{output_folder}/imgs/{name}').create()

        for i in range(downsample_rgb.shape[0]):
            tf.keras.utils.save_img(f'{output_folder}/imgs/{name}/{name}_{i}.jpg',downsample_rgb[i])
        

        




if __name__ == "__main__":
    preprocess()