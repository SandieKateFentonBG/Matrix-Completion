class model_results:

    def __init__(self, model, case_study, tr_losses, te_losses, rmse_te_losses, te_accuracies, best_val_acc, tr_accuracies):

        # Data split
        self.model = model
        self.case_study = case_study
        self.tr_losses = tr_losses
        self.te_losses = te_losses
        self.rmse_te_losses = rmse_te_losses
        self.te_accuracies = te_accuracies
        self.best_val_acc = best_val_acc
        self.tr_accuracies = tr_accuracies

def model_results_print(model_results, mystudy, folder=None, new_folder=False, VISU = False):

    if new_folder:
        from s10_export_resultsnew import mkdir_p
        mkdir_p(mystudy.output_path)

    for k, v in model_results.__dict__.items():
        if VISU:
            print(' ', k, ' : ', v)
        if folder:
            with open(mystudy.output_path + mystudy.reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()

