import numpy as np
import pandas as pd
import yfinance as yf
import nltk
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob, Word

model = load_model('E:\Stock_Market_Prediction_ML\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

# Read the CSV file
ndf = pd.read_csv('india-news-headlines.csv', parse_dates=[0], infer_datetime_format=True, usecols=["publish_date", "headline_text"])

# Rename the column
ndf = ndf.rename(columns={"publish_date": "Date"})

start_date = pd.to_datetime('2019-06-30')
end_date = pd.to_datetime('2020-06-30')
ndf=ndf.loc[(ndf['Date'] > start_date) & (ndf['Date'] < end_date)]

ndf=ndf.reset_index()

ndf=ndf.drop("index",axis=1)

# Dropping duplicates by grouping the same dates.
ndf['headline_text'] = ndf.groupby(['Date']).transform(lambda x : ' '.join(x)) 
ndf = ndf.drop_duplicates() 
ndf.reset_index(inplace = True, drop = True)

# uppercase-lowercase conversion
ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# numbers
ndf['headline_text'] = ndf['headline_text'].str.replace('\d','')

#stopwords

nltk.download('stopwords')

from nltk.corpus import stopwords
sw = stopwords.words('english')
ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

## Deletion of sparse.
delete = pd.Series(' '.join(ndf['headline_text']).split()).value_counts()[-1000:]
ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

nltk.download('wordnet')

ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#Functions to get the subjectivity and polarity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return  TextBlob(text).sentiment.polarity

ndf['Subjectivity'] = ndf['headline_text'].apply(getSubjectivity)
ndf['Polarity'] = ndf['headline_text'].apply(getPolarity)
ndf.head()

nltk.download('vader_lexicon')

#Adding sentiment score to ndf by using SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

ndf['Compound'] = [sia.polarity_scores(v)['compound'] for v in ndf['headline_text']]
ndf['Negative'] = [sia.polarity_scores(v)['neg'] for v in ndf['headline_text']]
ndf['Neutral'] = [sia.polarity_scores(v)['neu'] for v in ndf['headline_text']]
ndf['Positive'] = [sia.polarity_scores(v)['pos'] for v in ndf['headline_text']]
ndf[0:5]

df_merge = pd.merge(data, ndf, how='inner', on='Date')

df_final = df_merge[['Close','Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral' ,'Positive']]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df_scaled = pd.DataFrame(sc.fit_transform(df_final))
df_scaled.columns = df_final.columns
df_scaled.index = df_final.index
df_scaled.head()

fig5 = plt.figure(figsize=(8,6))
# Plotting polarity
Polarity = df_merge['Polarity']
Positive = df_merge['Positive']
Negative = df_merge['Negative']
plt.plot(Polarity, 'r',label='Polarity')
plt.plot(Positive, 'b',label='Positive')
plt.plot(Negative, 'g',label='Negative')
plt.xlabel('Date')
plt.ylabel('Sentiment Analysis')
plt.legend()
plt.switch_backend('Agg')  # Set backend to Agg
plt.show()
st.pyplot(fig5)
