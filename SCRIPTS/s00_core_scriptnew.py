
from s09_regularization_hypertuningnew import *
from SCRIPTS.v2.s10_export_results import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210523"
repository = "C:/Users/sfenton/Code/Repositories/"

#Parameter options
param_dict = dict()
param_dict['regularization_term']=[0.001, 0.01, 0.1, 1, 100, 1000]
param_dict['hidden_dim']=[10, 20, 40, 80, 100, 200, 300, 400, 500]
param_dict['i_u_study']=0
param_dict['database_group_split']=5
param_dict['selected_group']=0
param_dict['batch_size']=8
param_dict['input_dim']=[7699, 158]
param_dict['num_epochs']=5
param_dict['learning_rate']=0.01
param_dict['device']='cuda:0'


m_c = matrix_completion(project, database, date, repository, i_u_study = 0, database_group_split = 5, selected_group = 0,
                  batch_size = 8, input_dim = [7699, 158], hidden_dim = 500, num_epochs = 5, learning_rate = 0.001,
                  regularization_term = 0.01, device = 'cuda:0', VISU = True, new_folder=False)

AE_list = run_model_architectures(param_dict, project, database, date, repository,VISU = True, new_folder = True)
res = tune_model_architectures(x0 = [0.01, 500])
print(res.x)

export_results_as_csv(AE_list)

#plot_x_y_graph(x_list, y_list, x_label, y_label, title=None, folder=None, VISU=False)
#plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=False)
#plot_results(tr_losses,te_losses, reference = None, folder = None, VISU = False)




