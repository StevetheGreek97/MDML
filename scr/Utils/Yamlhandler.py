import yaml

class YamlHandler:
    """
    A class for handling YAML files
    """
    def __init__(self, file_path):
        """
        Constructor for the class

        Parameters
        ----------
         file_path : path of the YAML file
        """
        self.file_path = file_path

    def read_yaml(self):
        """
        Function for reading from a YAML file

        Returns
        -------
        data in the YAML file
        """
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def write_yaml(self, data):
        """
        Function for writing to a YAML file

        Parameters
        ----------
        data : data to be written to the file

        Returns
        -------
        Nothing
        """
        with open(self.file_path, 'w') as file:
            yaml.dump(data, file)

    
    def update_yaml(self, new_key, new_value):
        """
        Function for updating the YAML file
        
        Parameters
        ----------
        new_key: any
        The key of the dictionary

        new_value: any
        The value of the key for the dictonary
        
        Returns
        -------
        Nothing
        """
        # read data from the YAML file
        data = self.read_yaml()

        # update the data
        data[new_key] = new_value

        # write the updated data to the YAML file
        self.write_yaml(data)

    def read_key(self, key):
        """
        Function that reads and return a specific value given the key
        
        Paramenters
        ------------
        key : str
        The key for the dictinary to return the value of
        
        Returns
        --------
        The value of the key specified
        """

        data = self.read_yaml()
        return data[key]

    
def main():
    # example code

    # initialize the class with the path of the YAML file
    yaml_handler = YamlHandler('path/to/file.yml')

    # read data from the YAML file
    data = yaml_handler.read_yaml()
    print(data)

    yaml_handler.update_yaml('a', 'b')

    # read the data again to check if it was written correctly
    data = yaml_handler.read_yaml()
    print(data)

if __name__ == "__main__":
    main()

