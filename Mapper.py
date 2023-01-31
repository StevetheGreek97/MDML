import matplotlib.pyplot as plt
import os

def plot_saliencny_map(sal_map, title, path_to_save):
    """
    This function takes in a saliency map and saves the map
    as an image.

    python

    Parameters
    ----------
    sal_map: numpy.ndarray
        A 2D array representing the saliency map
    title: str
        The title for the saliency map
    path_to_save: str
        The path to save the saliency map

    """
    plt.title(title)
    plt.imshow(sal_map, cmap= 'gray')
    plt.clim(0, 1)
    plt.savefig(path_to_save)

def get_image_paths(path):
    """
    Generates the full path of each `.jpg` file in the specified `path`.

    Parameters:
        path (str): The path to the directory containing the image files.

    Yields:
        str: The full path to each `.jpg` file in the specified `path`.
    """
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.endswith('.jpg'):
            yield filename.path

def write_to_file(filename, text):
    """
    This function writes text to a file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    text : str
        The text to write to the file.

    Returns
    -------
    None
    """
    with open(filename, 'w') as file:
        file.write(text)

def find_res_index(state, ca):
    """
    This function takes in a state and a list of indices of
    Calcium atoms and returns a string of residues associated
    with those indices.

    markdown

    Parameters
    ----------
    state: mdtoolkit.universe.Universe
        The state that needs to be processed
    ca: List[int]
        Indices of Calcium atoms

    Returns
    -------
    str
        A string of the important residues separated by comma

    """
    index = 0 
    res_str = ''

    for atom in state.atoms:
        if atom.name == 'CA':
            index += 1
            if index in ca:
                res = str(atom.residue)
                res = res.split(' ')[-1][:-1]
                res_str += res
                res_str += ', '

    return res_str[:-2]

if __name__ == "__main__":
    pass