from Yamlhandler import YamlHandler as yamlh
from TrajectoryTools import LoadTrajectories
import tensorflow.keras as keras
import numpy as np
import Vision as vis
from setup import decode
from Retrace import plot_saliencny_map, get_image_paths, find_res_index


def mapping():
    """
    This function is used to compute the saliency map for a given set of images.
    It also saves the saliency map in a `.jpg` file and the important residues in a `.txt` file.
    """
    confkeys = yamlh('config.yml') 
    date = confkeys.read_key('date')
    save_path = confkeys.read_key('savepath')
    path = confkeys.read_key('masterpath')

    # Load data
    data  = LoadTrajectories().traj_from_dir(path + 'data/')
    print(data)

    # Load model
    try:
        model =  keras.models.load_model(f"{save_path}/output/models/{date}.h5")
    except OSError:
        print(f'The timestamp in the config has been changed.\
            {save_path}/output/models/{date}.h5')

    # Initialize saliency map
    sal_map = np.empty(shape = (
                confkeys.read_key('height'), 
                confkeys.read_key('lenght')
    ))

    # Iterate over the items in the data
    for name, universe in data.items():
        
        img_path = f"{save_path}/output/imgs/{name}"
        saving_path_per_sate = f"{save_path}/output/results/{name}_{date}"

        print(f'Computing saliency map for {name}...')
        for img in get_image_paths(img_path):
            sal_map += vis.SaliencyMap(model,decode(img)).gradient_saliency_map()

        # Normalize saliency map
        sal_map = sal_map / confkeys.read_key('downsample_to')

        # Save saliency map
        print('Saving..')
        plot_saliencny_map(sal_map, name, f"{saving_path_per_sate}.jpg")

        # Update the tempfactors of the residues
        for residue, r_value in zip(universe.select_atoms('protein').residues,np.mean(sal_map, axis =0)):
            residue.atoms.tempfactors = r_value
        print('Creating .pdb file')
        universe.select_atoms("protein").write(f"{saving_path_per_sate}.pdb") 

        # Find important residues
        mask = np.where(np.mean(sal_map, axis = 0) > 0.16)[0]
        important_res = find_res_index(universe, mask) 

        # Write to .txt file
        print('Writting to .txt file...')
        write_to_file(f'{saving_path_per_sate}.txt', important_res)


    


if __name__ == "__main__":
    mapping()