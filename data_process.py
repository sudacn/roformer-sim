import numpy as np 
import pandas as pd
import ipdb 
from config import Config
from os import read
from sklearn.utils import shuffle
from config import Config

class DataProcess(Config):
    def __init__(self) -> None:
        df = self.read_file(Config.csv_path)
        self.data_devide(df)

    def read_file(self, path):
        data_frame = pd.read_csv(path,nrows=15000)
        return data_frame

    def data_devide(self, data_frame):
        data_frame = shuffle(data_frame)
        train = data_frame.loc[:13000]
        dev = data_frame[13000:14000]
        test  = data_frame[14000:15000]
        self.del_index_save_csv(train,Config.train_path)
        self.del_index_save_csv(dev,Config.dev_path)
        self.del_index_save_csv(test,Config.test_path)


    def del_index_save_csv(self, df, path):
        del df['Unnamed: 0']
        df.reset_index(drop=True, inplace=True,)
        df.to_csv(path, index=False, encoding="utf_8_sig")
        print(path,"done!")
        

DataProcess()