import numpy as np
import pandas as pd
import torch

path = '/DATA/JesterDataset4/JesterDataset4.csv'

def open_csv_at_given_line(filename, first_line=0, delimiter=';'):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader

path = '/DATA/JesterDataset4/JesterDataset4.csv'

def quantitative_str_feature_to_float(string):
    """
    input : decimal number in string with "," for decimal separation
    output : decimal number in float with "." for decimal separation
    """

    try:
        return float(string.replace(',', '.'))
    except:
        print(string, ": this should be a number")
        return False


def string_dict_to_number_dict(index_dict,feature_dict, qualitative_features, quantitative_features ):
    number_dict = dict()
    for ql_feature in qualitative_features:
        number_dict[ql_feature] = []
        for ql_value in feature_dict[ql_feature]:
            number_dict[ql_feature].append(qualitative_str_feature_to_int(index_dict, ql_feature, ql_value))
    for qn_feature in quantitative_features:
        number_dict[qn_feature] = []
        for qn_value in feature_dict[qn_feature]:
            number_dict[qn_feature].append(quantitative_str_feature_to_float(qn_value))
    return number_dict

df = pd.read_csv(path, delimiter=';')
df.values
users_data = np.genfromtxt(path, delimiter=';')
users_tensor = torch.tensor(users_data)
items_tensor = torch.tensor()
for u in range(len(users_data)):  # 1699
    item_data = []
    for l in range(1, len(users_data[0])):
        item_data.append(users_data[u][l])
    items_tensor.append(item_data)
#print('i', type(items_data), len(items_data), len(items_data[7]), items_data[7])

"items_tensor = torch.tensor(items_data)

print(items_tensor[7])
print(user_tensor[0])
"""
df = pd.read_csv(path, delimiter=';')
df.values
users_data = np.genfromtxt(path, delimiter=';')
items_data = []
for u in range(len(users_data)):  # 1699
    item_data = []
    for l in range(1, len(users_data[0])):
        item_data.append(users_data[u][l])
    items_data.append(item_data)
print('i', type(items_data), len(items_data), len(items_data[7]), items_data[7])

items_tensor = torch.tensor(items_data)
user_tensor = torch.tensor(users_data)
print(items_tensor[7])
print(user_tensor[0])

labels = ['user_ratings']
for i in range(len(user_data[0]) - 1):
    labels.append(i)
dico = dict()
for l in range(len(labels)):  # 159+1
    dico[labels[l]] = []
for u in range(len(user_data)):  # 1699
    for l in range(len(labels)):
        dico[labels[l]].append(user_data[u][l])

print('labels', len(labels), labels)
print('keys', len(dico.keys()), dico.keys())
print('value', len(dico[labels[1]]), dico[labels[1]])
print('count', len(dico[labels[0]]), dico[labels[0]])


df = pd.read_csv(path, delimiter=';')
df.values
user_data = np.genfromtxt(path, delimiter=';')
labels = ['user_ratings']
for i in range(len(user_data[0])-1):
    labels.append(i)
dico = dict()
for l in range(len(labels)):  # 159+1
    dico[labels[l]] = []
for u in range(len(user_data)):  # 1699
    for l in range(len(labels)):
        dico[labels[l]].append(user_data[u][l])
        
        
print('labels', len(labels), labels)
print('keys', len(dico.keys()), dico.keys())
print('value', len(dico[labels[1]]), dico[labels[1]])
print('count', len(dico[labels[0]]), dico[labels[0]])

for elem in dico[labels[0]]:
    if elem == 0.0:
        print('no rating')
#print(dico)
"""
"""
def item_dict()
    df = pd.read_csv(path, delimiter=';')
    df.values
    user_data = np.genfromtxt(path, delimiter=';')
    labels = ['user_ratings'] + ['joke'+ str(i) ]
    for i in range(len(user_data[0])):  
        labels.append(i)
    dico = dict()
    for l in range(len(labels)): #159+1
        dico[labels[l]] = []
    for u in range(len(user_data)): #1699
        for l in range(len(labels)):
            dico[labels[l]].append(user_data[u][l])
            
    
            dico(l) =
        for j in range(len(X_values)):  # 80
            for i in range(len(X_values[0])):  # 12
                dico[x_labels[i]].append(X_values[j][i])
        dico['joke'+str(i)] = []

def index_dict_from_csv(filename, first_line=0, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    CST = dict()
    labels = ['user_ratings']
    for i in range(len(user_data[0])):  # 12
        dico['joke'+str(i)] = []


    for line in reader:
        for f in FEATURES_NAMES:
            index = header.index(f)
            if line[index] not in CST[f]:
                CST[f].append(line[index])
    return CST

def build_dictionary(X_values, y_values, x_labels, y_labels):  # TODO : shorten
    dico = dict()
    for i in range(len(X_values[0])):  # 12
        dico[x_labels[i]]=[]
    for i in range(len(y_values[0])):  # 12
        dico[y_labels[i]] = []
    for j in range(len(X_values)): #80
        for i in range(len(X_values[0])): #12
            dico[x_labels[i]].append(X_values[j][i])
        for k in range(len(y_values[0])):
            dico[y_labels[k]].append(y_values[j][k])
    return dico
"""