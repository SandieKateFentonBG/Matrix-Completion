class Autorec_model:

    def __init__(self, Autorec_data, Autorec_parameters, net):
        self.Autorec_data = Autorec_data
        self.Autorec_parameters = Autorec_parameters
        self.Autorec_net = Autorec_net

        self.reference = 'group_' + str(Autorec_data.selected_group) + '_h_dim_' + str(Autorec_parameters.hidden_dim) \
                         + '_regul_' + str(Autorec_parameters.regularization_term*1000)


