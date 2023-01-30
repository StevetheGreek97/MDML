import os

class CreateFolders:
    """
    Class for creating a folder and subfolders within it.

    Attributes:
        folder_name (str): Name of the main folder to be created.
    """

    def __init__(self, folder_name):
        """
        Initialize the main folder name.

        Args:
            folder_name (str): Name of the main folder to be created.
        """
        self.folder_name = folder_name

    def create(self):
        """
        Create the main folder if it doesn't already exist.
        """
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def create_subfolders(self):
        """
        Create subfolders within the main folder if they don't already exist.
        """
        # List of subfolder names to be created
        subfolder_names = ['results', 'performance', 'imgs', 'models']

        # Create each subfolder
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(self.folder_name, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)




if __name__ == "__main__":
    CreateFolders('output').create()

