import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import homogeneity_score
from sklearn.preprocessing import normalize
from memory_profiler import memory_usage

import pandas as pd
import numpy as np

class DatasetLoader():
    def __init__(self):
        self.df=None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.numbers_of_classes = None
        self.classes=None

    def read_dataset(self,file_path,separator=',',class_path=None):
        df = pd.read_csv(file_path, sep=separator)
        self.X = df.copy()
        if class_path!=None:
            df['class'] = pd.read_csv(class_path)
            self.y = df['class']
            self.classes=df['class'].unique()
            self.numbers_of_classes = len(self.classes)
        self.df=df


    def normalize(self,features_to_normalize=None): #features_to_normalize is a list of index
        if features_to_normalize!=None:
            return normalize(self.X[:,features_to_normalize[0]:features_to_normalize[1]])

    def select_features(self,features): #features is a list of features [feature1,feature2,...] or [:156]
        self.X=self.df.iloc[features]
    
    def select_classes(self,classes): #classes is a list of classes [class1,class2,...] with len(classes) = len of dataset
        self.y = classes
        self.classes=np.unique(classes)
        self.numbers_of_classes = len(self.classes)
        
    def split_dataset(self,test_size=0.2): #slit dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
    
    def split_dataset_class(self,class_to_group,random_state=1): #split dataset into train and test based on class
        # Initialize empty lists to store merged sets
        X_train_merged, X_test_merged, y_train_merged, y_test_merged = [], [], [], []
        for i in class_to_group.values():
            df_temp=self.df[self.df['class'].isin(i)]
            X_train, X_test, y_train, y_test = train_test_split(df_temp.iloc[:,:-1], df_temp['class'], test_size=0.2,random_state=random_state)
            X_train_merged.append(X_train)
            X_test_merged.append(X_test)
            y_train_merged.append(y_train)
            y_test_merged.append(y_test)

        # Merge sets
        return X_train_merged, X_test_merged, y_train_merged, y_test_merged
    
    def split_dataset_data(self,n,random_state=1): #split dataset by data size
        X_train_merged,X_test_merged,y_train_merged,y_test_merged=[],[],[],[]
        df_copy=self.df.copy()
        df_copy.pop('class')
        skf=StratifiedKFold(n_splits=n,shuffle=True,random_state=random_state)
        skf.get_n_splits(self.X_train,self.y_train)
        for i,(train_index, test_index) in enumerate(skf.split(self.X, self.y)):
            X_train_fold=df_copy.iloc[train_index]
            Y_train_fold=self.y[train_index]
            X_train_merged.append(X_train_fold)
            y_train_merged.append(Y_train_fold)

            X_test_fold=df_copy.iloc[test_index]
            Y_test_fold=self.y[test_index]
            X_test_merged.append(X_test_fold)
            y_test_merged.append(Y_test_fold)
        
        return X_train_merged,y_train_merged,X_test_merged,y_test_merged
    
class ModelLoader():
    def __init__(self):
        self.model=None
        self.initial_model=None
        self.optimizer_model=None
        self.optimal_params=None

    def set_optimal_params(self,optimal_params):
        self.optimal_params=optimal_params

    def set_model(self,model):
        self.model=model
        self.initial_model=model

    def reset_model(self):
        self.model=self.initial_model

    def optimize(self,X_train,y_train,cv=5,scoring='accuracy',n_iter=10):
        self.optimizer_model = RandomizedSearchCV(self.model,self.optimal_params,cv=cv,scoring=scoring,n_iter=n_iter)
        self.optimizer_model.fit(X_train,y_train)
        self.optimal_params = self.optimizer_model.best_params_

    def fit_train(self,X_train,y_train):
        return memory_usage(( self.model.fit, (X_train,y_train), {}), retval=True)

    def partial_fit_train(self,X_train,y_train,classes): # IF model compatible with partial_fit
        print("Partial fit")
        return memory_usage(( self.model.partial_fit, (X_train,y_train,classes), {}), retval=True)
        
    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def score(self,X_test,y_test):
        return self.model.score(X_test,y_test)

    def purity(self,y_test,y_pred):
        return homogeneity_score(y_test,y_pred)
