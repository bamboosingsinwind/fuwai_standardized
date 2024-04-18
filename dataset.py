import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import scale,MinMaxScaler
# import scipy.signal as sig
from scipy import signal as sig_fun

df = pd.read_excel("/home/fwyy/signaltest.xlsx")

class MyDataset(Dataset):
    
    def __init__(self,path,df,data_type):
        super(MyDataset,self).__init__()
        self.path = path
        self.data_type = data_type
        if data_type == "train":
            self.file_list = df[df["group"]==1]
            
        elif data_type == "valid":
            self.file_list = df[df["group"]==2]
            
        elif data_type == "test":
            self.file_list = df[df["group"]==3]
            
        else:
            print("error")
        self.file_list.reset_index(drop=True,inplace=True)

    def __getitem__(self,idx):
        file_path = self.path + self.file_list.loc[idx,"信号数据文件名"].replace("xml","csv")
        feat = []
        gender = self.file_list.loc[idx,"性别"]
        age =  self.file_list.loc[idx,"年龄"]/100
        feat += [gender,age]
        
        feat_plus = torch.FloatTensor(feat)
        df = pd.read_csv(file_path)
        df_lead = df[["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]] #12leads
        # df_lead = df[["I","II","III","AVR","AVL","AVF"]]#6leads
        sig = df_lead.values.T
       
        if self.data_type == "train":
            b,a = sig_fun.butter(1,0.5/250,"highpass")
            sig = sig_fun.filtfilt(b,a,sig,axis=1)
            b,a = sig_fun.butter(8,49/250,"lowpass")
            sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
            sig = sig_fun.medfilt(sig,(1,3))
        else:
            b,a = sig_fun.butter(1,0.5/250,"highpass")
            sig = sig_fun.filtfilt(b,a,sig,axis=1)
            b,a = sig_fun.butter(8,35/250,"lowpass")
            sig = sig_fun.filtfilt(b,a,sig,axis=1)#[:,0]
            sig = sig_fun.medfilt(sig,(1,3))
        sig = scale(sig,axis=1)
        # minmax_scaler = MinMaxScaler()
        # lead = minmax_scaler.fit_transform(df_lead).T
        # sig = sig[0:1,:]  #I lead
        
        lead = torch.FloatTensor(np.copy(sig))#[:,:512]
        label = self.file_list.loc[idx,"Label"]#.values
        return lead,feat_plus,label
    def __len__(self):
        return len(self.file_list)
path = "/home/fwyy/ecg_signal/csv_all/"
train_set = MyDataset(path,df,"train") 
valid_set = MyDataset(path,df,"valid")
test_set = MyDataset(path,df,"test")

print(train_set[1])
print(train_set[1][0].shape)  
print(len(train_set))
print(len(valid_set))
print(len(test_set))
