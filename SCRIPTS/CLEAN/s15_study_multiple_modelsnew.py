
from s09_regularization_hypertuningnew import *
from s10_export_resultsnew import *
from mc_tr_15_load_save_functions import *
from mc_tr_12_result_graphs import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210525"
repository = "C:/Users/sfenton/Code/Repositories/"

#Parameter options
param_dict = dict()
param_dict['regularization_term'] = [0.001] #[0.001, 0.01, 0.1, 1, 100, 1000]
param_dict['hidden_dim']=[300, 400, 500] #[10, 20, 40, 80, 100, 200, 300, 400, 500]
param_dict['i_u_study']=0
param_dict['database_group_split']=2
param_dict['selected_group']=0
param_dict['batch_size']=8
param_dict['input_dim']=[7699, 158]
param_dict['num_epochs']=10
param_dict['learning_rate']=0.01
param_dict['device']='cuda:0'

# 2. MULTIPLE MODELS
AE_list = run_model_architectures(param_dict, project, database, date, repository,VISU = False, new_folder = True)
#res = tune_model_architectures(x0 = [0.01, 500])
#print(res.x)

#DISPLAY MODELS
#print(AE_list)

#PLOT TRENDS
#y_list= m_c_results.rmse_te_losses
#x_list = [i for i in range(len(y_list))]
#y_label = 'rmse_te_losses'
#x_label = 'epochs'
#plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=True)
#plot_results(m_c_results.tr_losses,m_c_results.te_losses, reference = None, folder = None, VISU = True)
#plot_x_y_graph(x_list, y_list, x_label, y_label, title=None, folder=None, VISU=True)

# SAVE MODEL
save_list_pth(AE_list)
my_export(AE_list)
#export_multiple_results_as_csv(AE_list)

# LOAD MODEL
#torch.load('C:/Users/sfenton/Code/Repositories/Matrix-Completion/RESULTS/210523_results/JesterDataset4.pth')
#load_list_pth('C:/Users/sfenton/Code/Repositories/Matrix-Completion/RESULTS/210523_results/JesterDataset4.pth') #doesn't work







#plot rmse_te_loss - hidden layers










