"""import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html

import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_nse = pd.read_csv("alldata.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%m-%d-%Y")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["close"][i]=data["close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("saved_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:]
valid['Predictions']=closing_price



df= pd.read_csv("stock_data.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        data":[
                            go.Scatter(
                                x=train.index,
                                y=valid["close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                )                
            ])                


        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])

])



@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure


if __name__=='__main__':
    app.run_server(debug=True)











"""
"""
import pandas as pd
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html

# Load the stock data from the CSV file
df = pd.read_csv("stock_data.csv")

# Filter the data for the specified ticker symbols
ticker_symbols = ["TSLA", "MSFT", "AAPL", "FB"]
df_filtered = df[df["Stock"].isin(ticker_symbols)]

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div(
    children=[
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Actual vs Predicted",
                    children=[
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="actual-predicted-graph",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=df_filtered[df_filtered["Stock"] == stock]["Date"],
                                                y=df_filtered[df_filtered["Stock"] == stock]["Close"],
                                                mode="markers",
                                                name=f"Actual {stock}",
                                            )
                                            for stock in ticker_symbols
                                        ],
                                        "layout": go.Layout(
                                            title="Actual vs Predicted Closing Prices",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Price"},
                                        ),
                                    },
                                )
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="yearly High and Close Prices",
                    children=[
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="stock-dropdown",
                                    options=[
                                        {"label": stock, "value": stock} for stock in ticker_symbols
                                    ],
                                    value=ticker_symbols[0],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="yearly-prices"),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(
    dash.dependencies.Output("yearly-prices", "figure"),
    [dash.dependencies.Input("stock-dropdown", "value")],
)
def update_yearly_prices(stock):
    # Filter the data for the selected stock
    df_stock = df_filtered[df_filtered["Stock"] == stock]
    
    # Group the data by year and calculate the yearly high and close prices
    df_yearly = df_stock.groupby(df_stock["Date"].str[:4]).agg({"High": "max", "Close": "last"})
    
    # Create the bar graph for yearly high and close prices
    figure = {
        "data": [
            go.Bar(name="High", x=df_yearly.index, y=df_yearly["High"]),
            go.Bar(name="Close", x=df_yearly.index, y=df_yearly["Close"]),
        ],
        "layout": go.Layout(
            title=f"yearly High and Close Prices for {stock}",
            xaxis={"title": "Year"},
            yaxis={"title": "Price"},
            barmode="group",
        ),
    }
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
    """

"""








"""

import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html

# Load the stock data from the CSV file
df = pd.read_csv("alldata.csv")

# Filter the data for the specified ticker symbols
ticker_symbols = ["TSLA", "MSFT", "AAPL", "AMAZON","WALMART","ZOOM","UBER","NETFLIX","GOOGLE",]
df_filtered = df[df["Stock"].isin(ticker_symbols)]

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div(
    children=[
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Actual vs Predicted",
                    children=[
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="actual-predicted-graph",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=df_filtered[df_filtered["Stock"] == stock]["Date"],
                                                y=df_filtered[df_filtered["Stock"] == stock]["Close"],
                                                mode="markers",
                                                name=f"Actual {stock}",
                                            )
                                            for stock in ticker_symbols
                                        ],
                                        "layout": go.Layout(
                                            title="Actual vs Predicted Closing Prices",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Price"},
                                        ),
                                    },
                                )
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="yearly High and Close Prices",
                    children=[
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="stock-dropdown",
                                    options=[
                                        {"label": stock, "value": stock} for stock in ticker_symbols
                                    ],
                                    value=ticker_symbols[0],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="yearly-prices"),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(
    dash.dependencies.Output("yearly-prices", "figure"),
    [dash.dependencies.Input("stock-dropdown", "value")],
)
def update_yearly_prices(stock):
    # Filter the data for the selected stock
    df_stock = df_filtered[df_filtered["Stock"] == stock]
    
    # Group the data by year and calculate the yearly high and close prices
    df_yearly = df_stock.groupby(df_stock["Date"].str[:4]).agg({"High": "max", "Close": "last"})
    
    # Create the bar graph for yearly high and close prices
    figure = {
        "data": [
            go.Bar(name="High", x=df_yearly.index, y=df_yearly["High"]),
            go.Bar(name="Close", x=df_yearly.index, y=df_yearly["Close"]),
        ],
        "layout": go.Layout(
            title=f"yearly High and Close Prices for {stock}",
            xaxis={"title": "Year"},
            yaxis={"title": "Price"},
            barmode="group",
        ),
    }
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)

"""
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the stock data from the CSV file
df = pd.read_csv("stock_data.csv")

# Filter the data for the specified ticker symbol
ticker_symbol = "TSLA"
df_filtered = df[df["Stock"] == ticker_symbol].copy()

# Prepare the data for LSTM modeling
df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])
df_filtered.set_index("Date", inplace=True)
df_filtered.sort_index(ascending=True, inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_filtered[["Close"]].values)

# Split the data into training and test sets
train_data = scaled_data[:-12]
test_data = scaled_data[-12:]

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Adjust as desired
X_train, y_train = create_sequences(train_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Generate predictions for the test set
inputs = scaled_data[len(scaled_data) - len(test_data) - seq_length :]
X_test, y_test = create_sequences(test_data, seq_length)
predicted_values = model.compile(X_test)
predicted_values = scaler.inverse_transform(predicted_values)

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div(
    children=[
        html.H1("Stock Price Prediction Dashboard", style={"textAlign": "center"}),
        dcc.Graph(
            id="prediction-graph",
            figure={
                "data": [
                    go.Scatter(
                        x=df_filtered.index[-len(test_data):],
                        y=df_filtered["Close"].values[-len(test_data):],
                        mode="lines",
                        name="Actual",
                    ),
                    go.Scatter(
                        x=df_filtered.index[-len(test_data):],
                        y=predicted_values.flatten(),
                        mode="lines",
                        name="Predicted",
                    ),
                ],
                "layout": go.Layout(
                    title="Actual vs Predicted Closing Prices",
                    xaxis={"title": "Date"},
                    yaxis={"title": "Closing Price"},
                ),
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
"""