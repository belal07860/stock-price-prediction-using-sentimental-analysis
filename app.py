import numpy as np
import pandas as pd
import yfinance as yf
import nltk
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob, Word

model = load_model('H:\Stock_Market_Prediction_ML\Stock Predictions Model.keras')

st.header('Market Price Anticipation')

stock =st.sidebar.text_input('Enter Stock Symnbol', 'GOOG')
#start = '2012-01-01'
#end = '2022-12-31'
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

data = yf.download(stock,start=start_date, end=end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title = stock)
st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
 st.header('Stock Data')
 data2 = data
 data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
 data2.dropna(inplace = True)
 st.write(data2)
 annual_return = data2['% Change'].mean()*252*100
 st.write('Annual Return is ',annual_return,'%') 
 stdev = np.std(data2['% Change'])*np.sqrt(252)
 st.write('Standard Deviation is ',stdev*100,'%')
 st.write('Risk Adj. Return is ',annual_return/(stdev*100))  


 #from alpha_vantage.fundamentaldata import FundamentalData
 #with fundamental_data:
  # key = 'YKYN75WXA96OK61I'
   #fd = FundamentalData(key,output_format = 'pandas')
   #st.subheader('Balance Sheet')
   #balance_sheet = fd.get_balance_sheet_annual(stock)[0]
   #bs = balance_sheet.T[2:]
   #bs.columns = list(balance_sheet.T.iloc[0])
   #st.write(bs)
   #st.subheader('Income Statement')
   #income_statement = fd.get_income_statement_annual(stock)[0]
   #is1 = income_statement.T[2:]
   #is1.columns = list(income_statement.T.iloc[0])
   #st.write(is1)
   #st.subheader('Cash Flow Statement')
   #cash_flow = fd.get_cash_flow_annual(stock)[0]
   #cf = cash_flow.T[2:]
   #cf.columns = list(cash_flow.T.iloc[0])
   #st.write(cf) 

#data = yf.download(stock, start ,end)

#st.subheader('Stock Data')
#st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

tab1, tab2, tab3 = st.tabs (["ma_50_days", "ma_100_days", "ma_200_days"])


with tab1:
 st.header("Price vs MA50") 
 ma_50_days = data.Close.rolling(50).mean()
 fig2 = plt.figure(figsize=(8,6))
 plt.plot(ma_50_days, 'r')
 plt.plot(data.Close, 'g')
 plt.show()
 st.pyplot(fig2)

with tab2:
 st.header('Price vs MA50 vs MA100')
 ma_100_days = data.Close.rolling(100).mean()
 fig3 = plt.figure(figsize=(8,6))
 plt.plot(ma_50_days, 'r')
 plt.plot(ma_100_days, 'b')
 plt.plot(data.Close, 'g')
 plt.show()
 st.pyplot(fig3)

with tab3:
 st.header('Price vs MA100 vs MA200')
 ma_200_days = data.Close.rolling(200).mean()
 fig4 = plt.figure(figsize=(8,6))
 plt.plot(ma_100_days, 'r')
 plt.plot(ma_200_days, 'b')
 plt.plot(data.Close, 'g')
 plt.show()
 st.pyplot(fig4)

 from stocknews import StockNews
 with news:
    st.header('News of (stock)')
    sn = StockNews(stock , save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
      st.subheader(f'News (i+1)')
      st.write(df_news['published'][i])
      st.write(df_news['title'][i])
      st.write(df_news['summary'][i])
      title_sentiment = df_news['sentiment_title'][i]
      st.write(f'Title Sentiment {title_sentiment}')
      news_sentiment = df_news['sentiment_summary'][i]
      st.write(f'News Sentiment {news_sentiment}')


# Function to create a Plotly graph
def create_polarity_graph(title_sentiments, news_sentiments):
    data = pd.DataFrame({
        'Article Index': list(range(1, len(title_sentiments) + 1)),
        'Title Sentiment': title_sentiments,
        'News Sentiment': news_sentiments
    })
    
    fig = px.line(data, x='Article Index', y=['Title Sentiment', 'News Sentiment'],
                  labels={'value': 'Sentiment Polarity', 'variable': 'Sentiment Type'},
                  title='Sentiment Polarity of Top 10 News Articles')
    return fig

# Streamlit app setup
st.title('Stock News Sentiment Analysis')

# User input for stock ticker
stock = st.text_input('Enter stock ticker:', 'AAPL')

if st.button('Analyze'):
    with st.spinner('Fetching news and analyzing sentiment...'):
        st.header(f'News of {stock}')
        sn = StockNews(stock, save_news=False)
        df_news = sn.read_rss()
        
        title_sentiments = []
        news_sentiments = []
        
        for i in range(10):
            st.subheader(f'News {i+1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment: {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment: {news_sentiment}')
            
            title_sentiments.append(title_sentiment)
            news_sentiments.append(news_sentiment)
        
        # Create and display the polarity graph
        fig = create_polarity_graph(title_sentiments, news_sentiments)
        st.plotly_chart(fig)




# Read the CSV file
#ndf = pd.read_csv('G:\Stock_Market_Prediction_ML\india-news-headlines.csv', parse_dates=[0], infer_datetime_format=True, usecols=["publish_date", "headline_text"])

# Rename the column
#ndf = ndf.rename(columns={"publish_date": "Date"})

#start_date = pd.to_datetime('2020-01-30')
#end_date = pd.to_datetime('2020-06-30')
#ndf=ndf.loc[(ndf['Date'] > start_date) & (ndf['Date'] < end_date)]

#ndf=ndf.reset_index()

#ndf=ndf.drop("index",axis=1)

# Dropping duplicates by grouping the same dates.
#ndf['headline_text'] = ndf.groupby(['Date']).transform(lambda x : ' '.join(x)) 
#ndf = ndf.drop_duplicates() 
#ndf.reset_index(inplace = True, drop = True)

 #uppercase-lowercase conversion
#ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# numbers
#ndf['headline_text'] = ndf['headline_text'].str.replace('\d','')

#stopwords

#nltk.download('stopwords')

#from nltk.corpus import stopwords
#sw = stopwords.words('english')
#ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

## Deletion of sparse.
#delete = pd.Series(' '.join(ndf['headline_text']).split()).value_counts()[-1000:]
#ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

#nltk.download('wordnet')

#ndf['headline_text'] = ndf['headline_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#Functions to get the subjectivity and polarity
#def getSubjectivity(text):
 #   return TextBlob(text).sentiment.subjectivity

#def getPolarity(text):
 #   return  TextBlob(text).sentiment.polarity

#ndf['Subjectivity'] = ndf['headline_text'].apply(getSubjectivity)
#ndf['Polarity'] = ndf['headline_text'].apply(getPolarity)
#ndf.head()

#nltk.download('vader_lexicon')

#Adding sentiment score to ndf by using SentimentIntensityAnalyzer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sia = SentimentIntensityAnalyzer()

#ndf['Compound'] = [sia.polarity_scores(v)['compound'] for v in ndf['headline_text']]
#ndf['Negative'] = [sia.polarity_scores(v)['neg'] for v in ndf['headline_text']]
#ndf['Neutral'] = [sia.polarity_scores(v)['neu'] for v in ndf['headline_text']]
#ndf['Positive'] = [sia.polarity_scores(v)['pos'] for v in ndf['headline_text']]
#ndf[0:5]

#df_merge = pd.merge(data, ndf, how='inner', on='Date')

#df_final = df_merge[['Close','Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral' ,'Positive']]

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#df_scaled = pd.DataFrame(sc.fit_transform(df_final))
#df_scaled.columns = df_final.columns
#df_scaled.index = df_final.index
#df_scaled.head()

tab1, tab2 = st.tabs (["Polarity_graph", "prediction_graph"])

#with tab1:
 #st.header('Polarity graph')
 #fig6 = plt.figure(figsize=(8,6))
 ##Plotting polarity
 #Polarity = df_merge['Polarity']
 #Positive = df_merge['Positive']
 #Negative = df_merge['Negative']
 #plt.plot(Polarity, 'r',label='Polarity')
 #plt.plot(Positive, 'b',label='Positive')
 #plt.plot(Negative, 'g',label='Negative')
 #plt.xlabel('Date')
 #plt.ylabel('Sentiment Analysis')
 #plt.legend()
 #plt.switch_backend('Agg')  # Set backend to Agg
 #plt.show()
 #st.pyplot(fig6)


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

with tab2:
 st.header('Original Price vs Predicted Price')
 fig5 = plt.figure(figsize=(8,6))
 plt.plot(predict, 'r', label='Original Price')
 plt.plot(y, 'g', label = 'Predicted Price')
 plt.xlabel('Time')
 plt.ylabel('Price')
 plt.show()
 st.pyplot(fig5)