def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def export_single_results_as_csv(AE_results):

    from itertools import zip_longest
    import csv

    path = AE_results.case_study.output_path
    ref = AE_results.case_study.date

    d = dict()

    for k, v in AE_results.case_study.__dict__.items():
        d[k] = v
        """
        d['autorec_te_losses'] = AE_list[i].te_losses
        d['rmse_te_losses'] = AE_list[i].rmse_te_losses
        d['te_accuracies'] = AE_list[i].te_accuracies
        d['autorec_tr_losses'] = AE_list[i].tr_losses
        d['tr_accuracies'] = AE_list[i].tr_accuracies
        d['best_val_acc'] = AE_list[i].best_val_acc
        for k, v in AE_results.__dict__.items():
            d[k] = v"""

    #export_data = zip_longest(*d, fillvalue='')

    with open(path + ref + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(d.keys())
        wr.writerow(d.values())

    myfile.close()

def export_multiple_results_as_csv(AE_list):

    from itertools import zip_longest
    import csv

    path = AE_list[0].case_study.output_path
    ref = AE_list[0].case_study.date

    for i in range(len(AE_list)):
        d = dict()
        for k, v in AE_list[i].case_study.__dict__.items():
            d[k] = v
        for k, v in AE_list[i].__dict__.items():
            d[k] = v
        export_data = zip_longest(*d, fillvalue='')

        with open(path + ref + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(d.keys())
            wr.writerows(export_data)
        myfile.close()


def my_export(AE_list):

    from itertools import zip_longest
    import csv

    path = AE_list[0].case_study.output_path
    ref = AE_list[0].case_study.date

    d = dict()
    d['group'] = []
    d['hidden_dim'] = []
    d['regularization'] = []
    d['autorec_te_losses'] = []
    d['rmse_te_losses'] = []
    d['te_accuracies'] = []
    d['autorec_tr_losses'] = []
    d['tr_accuracies'] = []
    d['best_val_acc'] = []
    export_data = []
    for i in range(len(AE_list)):
        d['group'].append(AE_list[i].case_study.selected_group)
        d['hidden_dim'].append(AE_list[i].case_study.hidden_dim)
        d['regularization'].append(AE_list[i].case_study.regularization_term)
        d['autorec_te_losses'].append(AE_list[i].te_losses)
        d['rmse_te_losses'].append(AE_list[i].rmse_te_losses)
        d['te_accuracies'].append(AE_list[i].te_accuracies)
        d['autorec_tr_losses'].append(AE_list[i].tr_losses)
        d['tr_accuracies'].append(AE_list[i].tr_accuracies)
        d['best_val_acc'].append(AE_list[i].best_val_acc)
        export_data.append(zip_longest(*d, fillvalue=''))

    with open(path + ref + '.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(d.keys())
        wr.writerows(export_data)
    myfile.close()


