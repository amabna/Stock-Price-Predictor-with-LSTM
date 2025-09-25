import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import streamlit as st
from PIL import Image
plt.style.use('fivethirtyeight')
###
st.write('''
# STOCK PRICE PREDICTOR

**TOPLEARN.COM**
''')
###
st.sidebar.header('INSERT DATA')
def data():
    symbol=st.sidebar.selectbox('select the symbol',['AMZN','FOOLAD','KHODRO','TSLA'])
    return symbol
###
def get_data(symbol):
    if symbol=='FOOLAD':
        df=pd.read_csv(r"C:\Users\F15\Desktop\Python\AI\Deep_Learning\stock_market_analysis\foolad.csv")
    elif symbol=='KHODRO':
        df=pd.read_csv(r"C:\Users\F15\Desktop\Python\AI\Deep_Learning\stock_market_analysis\khodro.csv")
    elif symbol=='AMZN':
        df=pd.read_csv(r"C:\Users\F15\Desktop\Python\AI\Deep_Learning\stock_market_analysis\amazon.csv")
    elif symbol=='TSLA':
        df=pd.read_csv(r"C:\Users\F15\Desktop\Python\AI\Deep_Learning\stock_market_analysis\tesla.csv")
    df=df.set_index(pd.DatetimeIndex(df['Date'].values))  
    return df
###
symbol =data()
df=get_data(symbol)
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*0.8)
scaler=MinMaxScaler(feature_range=(0,1))             # [X-X(min)] / [X(max)-X(min)]
scaled_data=scaler.fit_transform(dataset)
###
training_data=scaled_data[0:training_data_len , :]
xtrain=[]
ytrain=[]
n=60
for i in range(n,len(training_data)):
    xtrain.append(training_data[i-n:i , 0])
    ytrain.append(training_data[i,0])
xtrain , ytrain = np.array(xtrain),np.array(ytrain)
xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
###
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
###
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=32,epochs=10)
###
test_data=scaled_data[training_data_len - n : , :]
xtest=[]
ytest=dataset[training_data_len: , :]
for i in range(n,len(test_data)):
    xtest.append(test_data[i-n:i,0]) 
xtest=np.array(xtest)
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
###
prediction=model.predict(xtest)
prediction=scaler.inverse_transform(prediction)
rmse =np.sqrt(np.mean(((prediction- ytest)**2)))
st.header('RMSE: ')
st.success(rmse)
###
train=data[:training_data_len]
valid=data[training_data_len:]
valid['prediction']=prediction
###
plt.figure(figsize=(16,8))
plt.title('PREDICTOR')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train['Close'])
plt.plot(valid[['Close','prediction']])
plt.legend(['Train','Value','Prediction'])
plt.savefig('accuracy.png')
plt.show()
###
st.header('STOCK PREDICTOR ACCURACY : ')
newdf=data[-60:].values
snewdf=scaler.transform(newdf)
xtest=[]
xtest.append(snewdf)
xtest=np.array(xtest)
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
pred=model.predict(xtest)
pred=scaler.inverse_transform(pred)
st.header('predicted price for next day:')
st.success(pred)







