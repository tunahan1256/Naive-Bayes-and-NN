import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report

df=pd.read_csv('SUSY.csv')
df=df.iloc[:500000,:]
new_column_names = ['Label',
    'lepton 1 pT',
    'lepton 1 eta',
    'lepton 1 phi',
    'lepton 2 pT',
    'lepton 2 eta',
    'lepton 2 phi',
    'missing energy magnitude',
    'missing energy phi',
    'MET_rel',
    'axial MET',
    'M_R',
    'M_TR_2',
    'R',
    'MT2',
    'S_R',
    'M_Delta_R',
    'dPhi_r_b',
    'cos(theta_r1)'
]
df.columns=new_column_names

X,y=df.iloc[:,1:],df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(
      X.to_numpy(), y.to_numpy(), test_size=0.25, random_state=42)

print(df[df.columns[1:19]].describe())

########################################################################################################################
# #HISTOGRAMS
df.hist(figsize=(20, 20), bins=100, xlabelsize=8, ylabelsize=8,alpha=0.5, color='blue')


df_signals=df[df.iloc[:,0]==1]
df_background=df[df.iloc[:,0]==0]
for feature in new_column_names:
    plt.figure(figsize=(8, 6))
    plt.hist(df_signals[feature], bins=100, alpha=0.5, label='Signals', color='black',histtype='step')
    n, bins, patches =plt.hist(df_background[feature], bins=100, alpha=0, label='Background')
    plt.plot(bins[:-1] + 0.5 * (bins[1] - bins[0]), n, linestyle='dotted', color='red', linewidth=1)

    
    plt.xlabel(feature)
    
    plt.legend()

    
    plt.show()
    
######################################################################################################################################################
# #CORRELATIONS
# Compute the correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1 ,vmax=1, center=0,
            square=True, linewidths=.2, cbar_kws={"shrink": .5})

features = ['lepton 1 pT',
    'lepton 1 eta',
    'lepton 1 phi',
    'lepton 2 pT',
    'lepton 2 eta',
    'lepton 2 phi',
    'missing energy magnitude',
    'missing energy phi',
    'MET_rel',
    'axial MET',
    'M_R',
    'M_TR_2',
    'R',
    'MT2',
    'S_R',
    'M_Delta_R',
    'dPhi_r_b',
    'cos(theta_r1)'
]
pos=[]
neg=[]
neut=[]
for k in range(len(corr.iloc[:])):
  for kk in range(len(corr.iloc[:])):
    if corr.iloc[k,kk]>0.75 and kk>k:
      pos.append((k,kk))
    if corr.iloc[k,kk]<-0.75 and kk>k:
      neg.append((k,kk))
    if corr.iloc[k,kk]>-0.01 and corr.iloc[k,kk]<0.01 and kk>k:
        neut.append((k,kk))

for k in pos:
    print('Positive Correlated Features:'+features[k[0]]+' and '+ features[k[1]])
    
for k in neg:
    print('Negative Correlated Features:'+features[k[0]]+' and '+ features[k[1]])
for k in neut:
    print('Non-Correlated Features:'+features[k[0]]+' and '+ features[k[1]])
    

for k in pos:
     
    myplot=pd.DataFrame(X.iloc[:5000,k[0]])
    myplot=pd.concat([myplot,X.iloc[:5000,k[1]]],axis=1)

    fig=px.parallel_coordinates(myplot,width=600,height=600)
    fig.show()
for k in neut:
     
    myplot=pd.DataFrame(X.iloc[:1000,k[0]])
    myplot=pd.concat([myplot,X.iloc[:1000,k[1]]],axis=1)

    fig=px.parallel_coordinates(myplot,width=1000,height=1000)
    fig.show()
    break

##############################################################################################################################################
# #GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
selected_features=['lepton 1 pT','lepton 1 eta','lepton 2 phi','axial MET']

X_train1 , X_test1 , y_train1 , y_test1 = train_test_split(X,y,test_size=0.25,train_size=0.75)
X=df[selected_features].values
X_train2 , X_test2 , y_train2 , y_test2 = train_test_split(X,y,test_size=0.25,train_size=0.75)
gnb.fit(X_train1, y_train1)
predictions1=gnb.predict(X_test1)

gnb.fit(X_train2, y_train2)
predictions2=gnb.predict(X_test2)

print("Accuracy score with selected features:", accuracy_score(y_test1 , predictions1))
print("Accuracy score with all features:", accuracy_score(y_test2 , predictions2))

####################################################################################################################################################

# #NEURAL NETWORKS
def define_model(n_inputs=[18,]):
    model = Sequential()
    
    
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

X,y=df.iloc[:500000,1:],df.iloc[:500000,0]

X_train, X_test, y_train, y_test = train_test_split(
      X.to_numpy(), y.to_numpy(), test_size=0.25, random_state=42)

nn_model=define_model()
nn_model.fit(X_train, y_train, epochs=30,validation_split=0.2, batch_size=4096)
val_score = nn_model.evaluate(X_test, y_test)
print("Validation accuracy: ", val_score[1])
threshold = 0.5
predictions = (nn_model.predict(X_test) > threshold).astype(int)


cm = confusion_matrix(y_test, predictions)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, square=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test,predictions))