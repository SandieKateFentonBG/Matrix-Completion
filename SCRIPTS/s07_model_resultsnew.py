class model_results:

    def __init__(self, model, optimizer, case_study, tr_losses, te_losses, rmse_te_losses, te_accuracies, best_val_acc, tr_accuracies):

        # Data split
        self.model = model
        self.optimizer = optimizer
        self.case_study = case_study
        self.tr_losses = tr_losses
        self.te_losses = te_losses
        self.rmse_te_losses = rmse_te_losses
        self.te_accuracies = te_accuracies
        self.best_val_acc = best_val_acc
        self.tr_accuracies = tr_accuracies

def model_results_print(model_results, folder=None, new_folder=False, VISU = False):

    if new_folder:
        from SCRIPTS.CLEAN.s10_export_resultsnew import mkdir_p
        mkdir_p(model_results.case_study.output_path)

    for k, v in model_results.__dict__.items():
        if VISU:
            print(' ', k, ' : ', v)
        if folder:
            with open(model_results.case_study.output_path + model_results.case_study.reference + ".txt", 'a') as f:
                print(' ', k, ' : ', v, file=f)
            f.close()

