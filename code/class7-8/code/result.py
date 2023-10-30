import matplotlib.pyplot as plt
import statistics

epoch_1 =  [16.67,22.22,16.67,52.78,58.33,16.67,47.22,30.56,77.78,47.22]
epoch_10 = [100.00,97.22,100.00,100.00,100.00,97.22,97.22,100.00,97.22,97.22]
epoch_20 = [97.22,97.22,100.00,100.00,97.22,97.22,100.00,97.22,97.22,100.00]
epoch_30 = [97.22,97.22,97.22,97.22,97.22,97.22,97.22,97.22,97.22,97.22]
epoch_40 = [97.22,100.00,97.22,97.22,97.22,97.22,97.22,97.22,97.22,97.22]
epoch_1_mean = statistics.mean(epoch_1)
epoch_10_mean = statistics.mean(epoch_10)
epoch_20_mean = statistics.mean(epoch_20)
epoch_30_mean = statistics.mean(epoch_30)
epoch_40_mean = statistics.mean(epoch_40)
'''
learnRate = 0.01            
t2vRatio = 1.2            
t2vEpochs = 3             
batchSize = 2 
'''

lr_1 =      [30.56,16.67,22.22,22.22,16.67,16.67,16.67,16.67,16.67,16.67]
lr_01 =     [22.22,30.56,16.67,30.56,16.67,22.22,16.67,22.22,16.67,22.22]
lr_001 =    [100.00,97.22,100.00,100.00,100.00,97.22,97.22,100.00,97.22,97.22]
lr_0001 =   [30.56,50.00,22.22,33.33,47.22,47.22,16.67,30.56,72.22,52.78]
lr_1_mean = statistics.mean(lr_1)
lr_01_mean = statistics.mean(lr_01)
lr_001_mean = statistics.mean(lr_001)
lr_0001_mean = statistics.mean(lr_0001)
'''
maxepoch = 10     
t2vRatio = 1.2            
t2vEpochs = 3             
batchSize = 2 
'''
batchsize_1 = [100.00,97.22,97.22,94.44,100.00,97.22,97.22,97.22,97.22,100.00]
batchsize_2 = [100.00,97.22,100.00,100.00,100.00,97.22,97.22,100.00,97.22,97.22]
batchsize_4 = [100.00,97.22,97.22,97.22,97.22,100.00,97.22,97.22,97.22,100.00]
batchsize_8 = [97.22,97.22,100.00,97.22,100.00,97.22,97.22,97.22,97.22,97.22]
batchsize_1_mean = statistics.mean(batchsize_1)
batchsize_2_mean = statistics.mean(batchsize_2)
batchsize_4_mean = statistics.mean(batchsize_4)
batchsize_8_mean = statistics.mean(batchsize_8)
'''
maxepoch = 10     
t2vRatio = 1.2            
t2vEpochs = 3             
learnRate = 0.01
'''
# #折线图
# x = list(range(1, 11))

# plt.figure(figsize=(6, 4))
# plt.plot(x, epoch_1, label='epoch=1')
# plt.plot(x, epoch_10, label='epoch=10')
# plt.plot(x, epoch_20, label='epoch=20')
# plt.plot(x, epoch_30, label='epoch=30')
# plt.plot(x, epoch_40, label='epoch=40')

# plt.ylabel('Accuracy')
# plt.legend(bbox_to_anchor=(1,0), loc='lower right',frameon=False,fontsize=9)
# plt.title('Different Epoch Accuracy Graph')
# plt.tight_layout()


# plt.show()


#柱状图
plt.figure(figsize=(6, 4))

x = ['lr=1', 'lr=0.1', 'lr=0.01', 'lr=0.001']
y = [lr_1_mean,lr_01_mean,lr_001_mean,lr_0001_mean]

bars = plt.bar(x, y,width=0.4)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

plt.title("Different LearningRate Accuracy Graph")
plt.xlabel("LearningRate")
plt.ylabel("Accuracy(%)")

plt.show()