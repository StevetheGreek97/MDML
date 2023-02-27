# Import Libraries
import os
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisFromFunction
from Alligner import Allign

class LoadTrajectories:
    def get_files(self, path):
        """
        Returns a dictionary that maps subfolder names to lists of full file paths in those subfolders.
        
        The function looks for subfolders in the input path, and for each subfolder, it creates
        a list of the full file paths for the files in the subfolder that have the '.pdb' or '.xtc'
        extensions. The subfolder name is used as the key in the output dictionary, and the list of
        file paths is used as the value.
        
        Parameters
        ----------
        path : str
            The directory path to search for subfolders.
        
        Returns
        -------
        path_dict : dict
            A dictionary that maps subfolder names to lists of full file paths in those subfolders.
        """
        # Initialize an empty dictionary to store the file lists
        path_dict = {}
        
        # Iterate over the entries in the input path
        for entry in os.listdir(path):
            # Get the full path to the entry
            entry_path = os.path.join(path, entry)
            # If the entry is a directory, process it
            if os.path.isdir(entry_path):
                # Create a list of the full file paths for the files in the subfolder
                # that have the '.pdb' or '.xtc' extensions
                file_paths = [
                    os.path.join(entry_path, file)
                    for file in os.listdir(entry_path)
                    if file.endswith(('.pdb', '.xtc'))
                ]
                # Add the subfolder name and file list to the output dictionary
                path_dict[entry] = file_paths
        
        # Return the output dictionary

        return path_dict
    
    def traj_from_dir(self, path):
        """
        Returns a dictionary that maps subfolder names to trajectories created from the '.pdb' and '.xtc'
        files in those subfolders.
        
        The function uses the `get_files` method to find the '.pdb' and '.xtc' files in the subfolders
        of the input path, and it creates a `Trajectory` object for each pair of '.pdb' and '.xtc' files.
        The subfolder name is used as the key in the output dictionary, and the `Trajectory` object is
        used as the value.
        
        Parameters
        ----------
        path : str
            The directory path to search for subfolders.
        
        Returns
        -------
        traj_dict : dict
            A dictionary that maps subfolder names to trajectories created from the '.pdb' and '.xtc'
            files in those subfolders.
        """
        traj_dict = {}
        # Create the traj_dict dictionary using a dictionary comprehension
        for folder, files_paths in self.get_files(path).items():
            for file_path in files_paths:
                if file_path.endswith('.xtc'):
                       xtc = file_path
                elif file_path.endswith('.pdb'):
                       pdb = file_path
            traj_dict[folder] = mda.Universe(pdb, xtc)


        # Print a message indicating the number of trajectories found
        print(f'Found {len(traj_dict.keys())} trajectories:')
        # print(traj_dict)
        return traj_dict


class ParseTrajectory:
    """
    A class for parsing and analyzing molecular dynamics trajectories using MDAnalysis.

    Attributes:
        no (int): Keeps track of the number of instances of the class.
        ref (ndarray): Reference coordinates used for alignment.
    """
    no  = 0 
    ref = np.array([])

    def __init__(self, name, universe):
        """
        The constructor for the ParseTrajectories class.

        Args:
            name (str): The name of the instance.
            universe (MDAnalysis.core.groups.Universe): The universe object containing the trajectory data.
        """
        self. name = name
        self.universe = universe

        # Increment the class variable 'no' to keep track of the number of instances
        ParseTrajectory.no += 1

        # If this is the first instance, set the reference coordinates
        if ParseTrajectory.no == 1:
            ParseTrajectory.ref = universe.atoms.positions
        
    def get_coords(self):
        """
        Returns the coordinates of the atoms in the trajectory.

        Returns:
            coords (ndarray): The coordinates of the atoms in the trajectory.
        """
        # Use an AnalysisFromFunction to get the coordinates at each frame
        coords = AnalysisFromFunction(lambda ag: ag.positions.copy(), self.universe).run().results
        return coords['timeseries']
    
    def allign(self):
        """
        Aligns the trajectory to the reference coordinates.

        Returns:
            aligned_coords (ndarray): The aligned coordinates of the atoms in the trajectory.
        """
        # Use the Allign class to align the coordinates to the reference coordinates
        return Allign(ParseTrajectory.ref).transform(self.get_coords())

    def __repr__(self):
            """
            Returns a string representation of the class.

            Returns:
                representation (str): A string representation of the class.
            """
            return f'{self.__class__.__name__}(name={self.name}, universe={self.universe.atoms.positions.shape})'

def down_sample(trajectory, num_images):
        """
        Extract a specified number of sub-trajectories from a given trajectory.
        
        Parameters
        ----------
        trajectory : numpy.ndarray
            A 3D NumPy array containing the coordinates of the atoms in the trajectory.
        num_images : int
            The number of sub-trajectories to extract from the trajectory.
        
        Returns
        -------
        list of numpy.ndarray
            A list of NumPy arrays, each containing a sub-trajectory of the original trajectory.
        """
        # Calculate the number of frames in the trajectory
        num_frames = trajectory.shape[0]
        
        # Calculate the number of frames that will be excluded from the mini-trajectories
        mod = num_frames % num_images
        
        if mod != 0:
            # Exclude the first 'mod' frames from the trajectory
            trajectory = trajectory[mod:, :]
            print(f'WARNING: Unable to index full trajectory for {num_images} images, excluded first {mod} frames')
        
        # Extract the desired frames from the trajectory using NumPy array indexing and slicing
        mini_trajs = [trajectory[index:num_frames:num_images, :] for index in range(num_images)]
        
        # Return the resulting mini-trajectories as a list of NumPy arrays
        return np.array(mini_trajs)

def main():
    """
    Example code
    """
    
    data = LoadTrajectories().traj_from_dir('/path/to/data/')

    # create an instance of the class
    parse_trajectory = ParseTrajectory('my trajectory', data['my trajectory'])


    # align the coordinates to the reference coordinates
    aligned_coords = parse_trajectory.allign()


if __name__ == "__main__":
    main()