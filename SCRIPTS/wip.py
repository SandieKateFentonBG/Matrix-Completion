keys = ['x_train', 'x_test', 'x_val']
vals = [0, 1, 2]
data_dict = dict()

for k, v in zip(keys, vals):
    print(k, v)
    data_dict[k] = v
print(data_dict.items(), type(data_dict.items()))

for k, v in data_dict.items():
    print(k, v)
b = [[2,3],[1]]
#a = [[regul_list,hidden_list][i_u_study,database_group_split, selected_group, batch_size, input_dim, num_epochs, learning_rate, device ]]
print(len(b))
j,k = b[0]
print(j)
print(b[1])