from datetime import date
import os
from dataProcessor import *
from data import Data


def paramError(param):
    print("Invalid dataHandler parameter : " + param)


class DataHandler:
    def __init__(self, inputDim, databaseFilename='JesterDataset4/JesterDataset4',
                        projectPath=None, outputName=None,
                        itemBased=True, numberOfSubdata=5, batchSize=8):
        if type(inputDim) != list or len(inputDim) != 2:
            return paramError("inputDim")
        if type(itemBased) != bool:
            return paramError("itemBased")
        if type(numberOfSubdata) != int or numberOfSubdata < 1:
            return paramError("numberOfSubdata")
        if type(batch_size) != int or batch_size < 2 or batch_size > inputDim[int(itemBased)]:
            return paramError("batch_size")
        for param, strparam in [(outputName, "outputName"), (projectPath, "projectPath"), (databaseFilename, "databaseFilename")]:
            if param and type(param) != str:
                return paramError(strparam)
        self.inputDim = inputDim
        self.itemBased = itemBased
        self.numberOfSubdata = numberOfSubdata
        self.batchSize = batchSize
        path = projectPath or os.getcwd()
        self.input_path = projectPath + '/DATA/' + databaseFilename + ".csv"
        name = outputName or date.today().strftime("%Y/%m/%d")
        self.output_path = path + '/RESULTS/' + name + '_results/'
        self.currentSubdata = 0
        self.acquireData()

    def nextSubdata(self):
        self.currentSubdata = (self.currentSubdata + 1) % self.numberOfSubdata
        self.acquireData()

    def acquireData():
        subDatas = makeSubdata(self.numberOfSubdata, loadData(path))
        subData = subDatas[self.currentSubdata]
        studied_data = selectData(subData, self.itemBased)
        self.data = Data(makeLearningsGroups(studied_data),
                        makeDataLoader(x_train, x_test, x_val, self.batchSize))

    def printMe(self, title, save=False, show=False):
        if save and not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        for k, v in self.__dict__.items():
            if show:
                print(' ', k, ' : ', v)
            if save:
                print(' ', k, ' : ', v, file=open(self.output_path + title + ".txt", 'w+'))

    
 # "C:/Users/sfenton/Code/Repositories/Matrix-Completion"
        
        # i_u_study     0=item based, 1=user based 