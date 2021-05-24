
from s09_regularization_hypertuningnew import *
from s10_export_resultsnew import *
from s12_access_resultsnew import *
from s11_plot_resultsnew import *

#Reference
project = 'Matrix-Completion'
database = 'JesterDataset4/JesterDataset4.csv'
date = "210523"
repository = "C:/Users/sfenton/Code/Repositories/"

# 1. SINGLE MODEL
m_c_results = matrix_completion(project, database, date, repository, i_u_study = 0, database_group_split = 2, selected_group = 0,
                  batch_size = 8, input_dim = [7699, 158], hidden_dim = 500, num_epochs = 5, learning_rate = 0.001,
                  regularization_term = 0.01, device = 'cuda:0', VISU = False, new_folder=False)
#DISPLAY MODEL
#print(m_c_results)
#model_print(model)
#optimizer_print(optimizer)

#PLOT MODEL
#plot_results(m_c_results.tr_losses,m_c_results.te_losses, label_a = 'tr_losses', label_b = 'te_losses',
#             reference = None, folder = None, VISU = True)
#plot_all_results(m_c_results, y_label = 'AE Loss', x_label = 'Epochs', reference = None, folder = None, VISU = True)
#x_list, y_list = [i for i in range(len(y_list))], m_c_results.rmse_te_losses
#"x_label, y_label = 'rmse_te_losses' , 'epochs'
#plot_sns_graph(x_list, y_list, x_label, y_label, title=None, figure_size=(12,15), folder=None, plot=True)

# SAVE MODELS
#save_model(m_c_results, PATH = None, inference = False)
export_single_results_as_csv(m_c_results) #TODO : not working

# LOAD MODEL
#torch.load('C:/Users/sfenton/Code/Repositories/Matrix-Completion/RESULTS/210523_results/JesterDataset4.pth')
#load_model('C:/Users/sfenton/Code/Repositories/Matrix-Completion/RESULTS/210523_results/JesterDataset4.pth') #TODO : not working










#AE_list = run_model_architectures(param_dict, project, database, date, repository,VISU = True, new_folder = True)
#res = tune_model_architectures(x0 = [0.01, 500])
#print(res.x)







