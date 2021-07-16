import mne
import numpy as np
from mne.filter import notch_filter
from mne.io import read_raw_cnt, read_raw_fif
from mne.preprocessing import ICA
from mne import concatenate_raws
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.data as Data
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ReadFilterData(PATH, subject, filter_low_freg, filter_high_freg):
    if os.path.exists('fif/filter_data-raw.fif'):
        EEG = read_raw_fif('fif/filter_data-raw.fif')
        data = EEG.get_data()
        # print("data.size", data.size)
        # data.size 3783840
        # print(data.shape)
        # (60, 63064)
        events = mne.events_from_annotations(EEG)
        label_dict = {v : k for k, v in events[1].items()}
        for cnt, event in enumerate(events[0]):
            events[0][cnt][-1] = label_dict[event[-1]]
            events[0][cnt][-1] -= 1

        reject_criteria = dict(eeg=500e-6)       # 200 µV
        viepochs = mne.Epochs(EEG, events[0], tmin=-3, tmax=4,
                              reject=reject_criteria,
                              preload=True, baseline=(-3, -1))
        viepochs.save('fif/viepochs-epo.fif', overwrite=True)
    else:
        folder =PATH+'/rawData'

        fileNameList =[folder+'/'+subject+'{}.cnt'.format(i) for i in range(1, 13)]
        theEEGlist = []

        for f in fileNameList:
            # 1、读取文件
            theEEG = mne.io.read_raw_cnt(f, date_format='dd/mm/yy').load_data()
            # 2、陷波
            if notch_filter:
                theEEG.notch_filter(np.arange(50, 250, 50))
            theEEG.resample(200, npad="auto")
            # 3、剔除坏导
            bads = ['M1', 'M2', 'CB1', 'CB2', 'HEO', 'VEO', 'EKG', 'EMG']
            badChans = [i for i in bads if i in theEEG.info['ch_names']]
            theEEG.info['bads'] = badChans
            pick = mne.pick_types(theEEG.info, eeg=True, exclude='bads')

            # 4、滤波
            theEEG.filter(filter_low_freg, filter_high_freg)
            After_filter = f[:-4]+'-raw.fif'
            theEEG.save(After_filter, picks=pick, overwrite=True)
            theEEG = read_raw_fif(After_filter, preload=True)

            # 重定位
            montage = mne.channels.read_custom_montage("channel_location_60_neuroscan.locs")
            theEEG.set_montage(montage)

            # ICA
            ica = ICA(n_components=60, random_state=97, method='fastica')
            ica.fit(theEEG)
            eog_indices, eog_scores = ica.find_bads_eog(theEEG, 'FPZ')
            ica.exclude = eog_indices
            reconst_theEEG = theEEG.copy()
            ica.apply(reconst_theEEG)
            reconst_theEEG.save(After_filter, overwrite=True)
            theEEGlist.append(reconst_theEEG)

        # 连接原始实例
        EEG = concatenate_raws(theEEGlist).load_data()
        # {'eeg': 0.0005}
        reject_criteria = dict(eeg=500e-6)
        events = mne.events_from_annotations(EEG)
        label_dict = {v:k for k, v in events[1].items()}
        for cnt,event in enumerate(events[0]):
            events[0][cnt][-1] = label_dict[event[-1]]

        viepochs = mne.Epochs(EEG, events[0], tmin=-3, tmax=4, reject=reject_criteria,
                               preload=True, baseline=(-3, -1))
        if not os.path.exists('fif/filter_data.fif'):
            if not os.path.exists('fif'):
                os.makedirs('fif')
                EEG.save('fif/filter_data-raw.fif', overwrite=True)
            else:
                EEG.save('fif/filter_data-raw.fif',overwrite=True)
        viepochs.save('fif/voepochs-epo.fif', overwrite=True)
        data = EEG.get_data()
    return EEG, data, viepochs



PATH ="C:/Users/user/Desktop/code/Data/post-vi/tch"
subject ='tch'

EEG, data, viepochs = ReadFilterData(PATH, subject, 1., 50.)

X = viepochs.get_data()


Y = viepochs.events[:, -1]

X = X[:, :, 0:1184]
# (5,60, 1200)
# print("After shape", X.shape)
# 100、

kernels, chans, samples = 1, 60, 1184
# 训练集
X_train = X[0:200, ]
Y_train = Y[0:200]
# 验证集
X_val = X[200:240, ]
Y_val = Y[200:240]
# 测试集
X_test = X[240:, ]
Y_test = Y[240:]






# 将标签设为Longtensor并转为one-hot形式
train_label = torch.from_numpy(Y_train).to(torch.int64)
# one_hot_train = torch.zeros(5, 3).scatter_(1, train_label, 1)
one_hot_train = train_label.reshape(200, )

val_label = torch.from_numpy(Y_val).to(torch.int64)
one_hot_val = val_label.reshape(40, )
# one_hot_val = torch.zeros(5, 3).scatter_(1, val_label, 1)

test_label = torch.from_numpy(Y_test).to(torch.int64)
one_hot_test = test_label.reshape(48, )

# print("训练标签", one_hot_train)


# 将x变为tensor形式
train_X = torch.from_numpy(X_train)
val_X = torch.from_numpy(X_val)
test_X = torch.from_numpy(X_test)


# 统一变成4维
# train_X = train_X.reshape(train_X.shape[0], chans, samples,1)
train_X = train_X.reshape(train_X.shape[0], 1, chans, samples)
train_X = train_X.to(torch.float32)

val_X = val_X.reshape(val_X.shape[0], 1, chans, samples)
val_X = val_X.to(torch.float32)

test_X = test_X.reshape(test_X.shape[0], 1, chans, samples)
test_X = test_X.to(torch.float32)

# print("shape", train_X.shape)
# # shape torch.Size([5, 1, 60, 1200])
# print(train_X.shape)




# 方案1 EEGNET
class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters F1=自己设定的
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)/（F1，C，T）
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(1, 64),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((16 * 2100), classes_num)
        # mat1 and mat2 shapes cannot be multiplied (5x560 and 33600x3) 16*2100
        # mat1 and mat2 shapes cannot be multiplied (5x560 and 336x3) 16*21

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return F.softmax(x, dim=1), x  # return x for visualization
classname = 3
model = EEGNet(classes_num=3)
LR = 0.0005
Batch_size = 20
EPOCH = 10

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

dataset = Data.TensorDataset(train_X, one_hot_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=Batch_size, shuffle=True, num_workers=0)

# plt.ion()
# print('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))

# list_loss = []
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(dataloader):
#         output = model(b_x)[0]
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % 5 == 0:
#             test_out, last_layer = model(test_X)
#             pred_y = torch.max(test_out, 1)[1].data.numpy()
#             accuracy = float((pred_y == one_hot_test.data.numpy()).astype(int).sum())/float(test_label.size(0))
#             print("\033[0;31;40mstep\033[0m", step, "accuracy:", accuracy,"Loss:",loss)
#             list_loss.append(loss)
# print(list_loss)
#
# # torch.save(model.state_dict(), 'C:/Users/user/Desktop/code/tmp/model_parm1.pt')
# torch.save(model.state_dict(), 'tmp/model_parm1.pt')

model1 = EEGNet(classes_num=3)
model1.load_state_dict(torch.load('C:/Users/user/Desktop/code/tmp/model_parm1.pt'))
for i in range(150):
    val_out, last_layer = model1(val_X)
    pred_y = torch.max(val_out, 1)[1].data.numpy()
    print("真实值", one_hot_val)
    print("预测值", pred_y)
    accuracy = float((pred_y == one_hot_val.data.numpy()).astype(int).sum())/float(val_label.size(0))
    print("accuracy:",accuracy)













