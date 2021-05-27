import numpy as np
import torch

class Training_data:

        def __init__(self, i_u_study=0, database_group_split=5, selected_group=0, batch_size=8,input_dim=[7699, 158],
                     date='210525', project='Matrix-Completion', database='JesterDataset4/JesterDataset4.csv',
                     repository="C:/Users/sfenton/Code/Repositories/"):

            # Referencing
            self.date = date
            self.input_path = repository + str(project) + '/DATA/' + str(database)
            self.output_path = repository + project + '/RESULTS/' + date + '_results/'

            # Data split
            # focus of the study : { 0=item based, 1=user based }
            self.i_u_study = i_u_study
            # split database into data groups :
            self.database_group_split = database_group_split
            # select group to work on :
            self.selected_group = selected_group
            # iterate through data in batches :
            self.batch_size = batch_size
            # Dimension of input data :
            self.input_dim = input_dim  # ratings_per_item, ratings_per_user

            def __getitem__(self, key):
                return getattr(self, key)
                # items = mystudy.__dict__.items()

            def __setitem__(self, key, val):
                return setattr(self, key, val)


def studied_attribute(mystudy, attr, val):
    mystudy.__setitem__(attr, val)
    return attr, val


def mc_tr_data_print(mydata, reference, folder=None, new_folder=False, VISU=False):
    if new_folder:
        from mc_tr_05_Autorec_net import mkdir_p
        mkdir_p(mydata.output_path)

    for k, v in mydata.__dict__.items():
        if VISU:
            print(' ', k, ' : ', v)
        if folder:
            with open(mydata.output_path + reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()
