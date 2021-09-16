import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

class LinReg:
    
    def listFiles(self):
        import os
        files = os.listdir(".")
        print("FILES")
        for f in files:
            print(f)
            
    def getDir(self):
        import os
        print(os.getcwd())
    
    def getData(self):
        #curDir = "/rapids/notebooks/workspace/FullExample"
        curDir = "."
        self.X = np.load(curDir + '/data/MLindependent.npy',allow_pickle = True)
        self.y = np.load(curDir + '/data/MLdependent.npy',allow_pickle = True)
        self.XNames = np.load(curDir + '/data/MLindependentNames.npy',allow_pickle = True)
        self.yNames = np.load(curDir + '/data/MLdependentNames.npy',allow_pickle = True)
        return([self.XNames, self.yNames])
    
    def printNames(self):
        print("X: ", self.XNames)
        print("y: ", self.yNames)
        
    def splitData(self):
        #Splitting the data set into training and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size =1/3, random_state = 0)
        return([self.X_train, self.X_test, self.y_train, self.y_test])
        
    def formModel(self):
        #Fitting simple linear regression to training set
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)

    def makePredictions(self):
        #makes some predictions
        self.X_test = self.X_test.reshape(-1,1)  #had to add this line JWH
        ypred = self.regressor.predict(self.X_test)
        trainRscore = self.regressor.score(self.X_train, self.y_train)
        testRscore =  self.regressor.score(self.X_test, self.y_test)
        return([trainRscore, testRscore])

    def plotTrainingResults(self, localFlag = True):
        plt.scatter(self.X_train.tolist(), self.y_train.tolist(), color = 'red')
        pre = self.regressor.predict(self.X_train)
        plt.scatter(self.X_train.tolist(), pre.tolist(), color = 'blue')
        plt.title(self.XNames[0] + " vs " + self.yNames[0])
        plt.xlabel(self.XNames[0])
        plt.ylabel(self.yNames[0])
        if(localFlag == True):
            plt.show()
        else:
            plt.savefig("output/LinearRegressionTrainingResults.png")
    
    def plotTestResults(self,localFlag=True):
        
        plt.scatter(self.X_test.tolist(), self.y_test.tolist(), color = 'red')
        plt.scatter(self.X_train.tolist(), self.regressor.predict(self.X_train).tolist(), color = 'blue')
        plt.title(self.XNames[0] + " vs " + self.yNames[0] + " (Training Set)")
        plt.xlabel(self.XNames[0])
        plt.ylabel(self.yNames[0])
        if(localFlag == True):
            plt.show()
        else:
            plt.savefig("output/LinearRegressionTestResults.png")


# Local Run
if __name__ == "__main__":
    lin = LinReg()
    lin.getDir()
    lin.listFiles()
    lin.getData()
    lin.printNames()
    lin.splitData()
    lin.formModel()
    lin.makePredictions()
    lin.plotTrainingResults(localFlag=False)
    lin.plotTestResults(localFlag=False)
    print("Finished Run")

