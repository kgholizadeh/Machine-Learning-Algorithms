import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv("D:\Shuttle\shuttle-trn.txt", delimiter='\t', header=None)
inputs = df[[0,1,2,3,4,5,6,7,8]]
targets = df[9]
validation_size = 0.20
seed = 7
inputs, X_validation, targets, Y_validation = train_test_split(inputs, targets, test_size=validation_size, random_state=seed)
test_df = pd.read_csv("D:\Shuttle\shuttle-tst.txt", delimiter='\t', header=None)
test_x = test_df[[0,1,2,3,4,5,6,7,8]]
test_y = test_df[[9]]
print(targets.value_counts())
print(df.shape)
print(test_df.shape)
print(df.head(120))
print(test_df.head(120))
print(df.describe())
print(test_df.describe())
print(df.groupby(9).size())
print(test_df.groupby(9).size())
#df.plot(kind='box', subplots=True, layout=(11,11), sharex=False, sharey=False)
plt.show()
#test_df.hist()
# plt.show()
