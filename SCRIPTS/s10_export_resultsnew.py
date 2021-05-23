def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def export_results_as_csv(results_list):

    from itertools import zip_longest

    path = results_list[0].case_study.output_path
    ref = results_list[0].case_study.date

    d = dict()

    for k, v in model_result.case_study.__dict__.items():
        d[k] = mydict
    for k, v in model_result.__dict__.items():
        d[k] = mydict
    export_data = zip_longest(*d, fillvalue='')

    with open(path + ref + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(mystudy.__dict__.keys())
        wr.writerows(export_data)

    myfile.close()


    """ 
    model_result.case_study.__dict__.items()
    model_result.tr_losses = tr_losses
    model_result.te_losses = te_losses
    model_result.rmse_te_losses = rmse_te_losses
    model_result.te_accuracies = te_accuracies
    model_result.best_val_acc = best_val_acc
    model_result.tr_accuracies = tr_accuracies"""

def export_results_as_csvold(path, res_dict, filename="results"):

    from itertools import zip_longest
    d = [res_dict[key] for key in res_dict.keys()]
    export_data = zip_longest(*d, fillvalue='')
    with open(path + filename + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(res_dict.keys())
        wr.writerows(export_data)
    myfile.close()
