import os


class DataHandler:

    def __init__(self, path, train_size=.7) -> None:
        self.train_size = train_size
        self.root_dir = path

    def folder_structure(self, dest_folders = ['train', 'test']):
        os.chdir(self.root_dir)
        


        for item in dest_folders:
            for i in range(len(dest_folders)):
                os.mkdir(f'mkdir -p {self.root_dir}/datasets/{item}/{str(i)}')
                

        # folder structure like this:   datasets/train/0
        #                               datasets/train/1
        #                               datasets/test/0
        #                               datasets/test/1
        #                               datasets/val/0
        #                               datasets/val/1

loader = DataHandler('C:/Users/Venati Himanth/OneDrive/Desktop/strive/python/DL/PyTorch/99. Live_coding/03. classification_project')

loader.folder_structure(loader.root_dir )



    








