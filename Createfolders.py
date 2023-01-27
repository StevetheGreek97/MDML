import os

class CreateFolders:
    """
    A class for creating an output folder and within that folder, four other folders namely, 
    'results', 'performance', 'imgs', 'models'
    """
    def __init__(self, output_folder):
        """
        Constructor for the class
        :param output_folder: name of the output folder
        """
        self.output_folder = output_folder
        # list of folders to be created within the output folder
        self.folders = ['results', 'performance', 'imgs', 'models']

        # check if output folder already exists, if not create it
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        
        # create the folders
        for folder in self.folders:
            os.makedirs(os.path.join(self.output_folder, folder))



if __name__ == "__main__":
    pass
    #CreateFolders('output')

