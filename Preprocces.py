from Createfolders import CreateFolders as cf
from Yamlhandler import YamlHandler as yamlh
from TrajectoryTools import LoadTrajectories, ParseTrajectories


def preprocess():
    # cf('output')
    path = yamlh('config.yml').read_key('masterpath')
    data  = LoadTrajectories().traj_from_dir(path + 'data/')

    for (name,  trajectory) in data.items():
        traj = ParseTrajectories(name, trajectory).allign()
        print(traj)




if __name__ == "__main__":
    preprocess()