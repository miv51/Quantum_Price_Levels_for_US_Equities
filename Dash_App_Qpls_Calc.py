
# DEPENDENCIES

#!pip install dash
#!pip install plotly
#!pip install yfinance
#!pip install pandas
#!pip install numpy

from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go

import yfinance
import pandas
import numpy

print('IMPORTS DONE')

candle_chart_1m = 'candle_chart_1min'
candle_chart_5m = 'candle_chart_5min'
candle_chart_1d = 'candle_chart_1day'

g_dist_plot = 'gaussian_distribution_chart'
tick_select = 'ticker_selection'
update_int = 'interval'

fps = 0.5
width = 700
height = 500

app = Dash(__name__)

app.layout = html.Div([
    html.Label('Ticker Symbol '),
    dcc.Input(id=tick_select, placeholder='AAPL', type='text', value=''),
    
    html.Div([
        dcc.Graph(id=candle_chart_1m, style={'display': 'inline-block'}),
        dcc.Graph(id=candle_chart_5m, style={'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id=candle_chart_1d, style={'display': 'inline-block'}),
        dcc.Graph(id=g_dist_plot, style={'display': 'inline-block'})
    ]),
    
    dcc.Interval(id=update_int, interval=int(1000 / fps)),
])

def create_candle_chart(candle_data, title, qpls=numpy.array([]), nearest=9999):
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('OHLC', 'Volume'), row_width=[0.3, 0.7])
    
    fig.add_trace(go.Candlestick(x=candle_data.index, open=candle_data['Open'],
                                 high=candle_data['High'], low=candle_data['Low'],
                                 close=candle_data['Close'], showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Bar(x=candle_data.index, y=candle_data['Volume'], showlegend=False), row=2, col=1)
    fig.update_layout(title=title, yaxis_title="Price (USD)", width=width, height=height)
    fig.update(layout_xaxis_rangeslider_visible=False)
    
    last_close = candle_data['Adj Close'].iloc[-1]
    
    qpls = list(zip(list(((qpls-last_close)**2)**0.5),list(qpls))) # sort them by distance to last_close
    
    qpls.sort()
    
    qplc = {'color':'rgba(64,64,255,0.5)'}
    rx = [candle_data.index[0], candle_data.index[-1]]
    
    for dist, qpl in qpls[:nearest]:
        fig.add_trace(go.Scatter(x=rx, y=2*[qpl], showlegend=False, line=qplc), row=1, col=1)
        
    return fig

def calc_qpls(daily_closing_prices):
    
    # using the past 2048 days as in the book (for a different security)
    # if there are <= 2048 days of data then use the number of available days minus one
    lookback = 2048 if len(daily_closing_prices) > 2048 else max(len(daily_closing_prices) - 1, 0)
    
    daily_relative_changes = (daily_closing_prices / daily_closing_prices.shift(1) - 1).iloc[-lookback:]
    
    mean = daily_relative_changes.mean()
    std = daily_relative_changes.std()
    
    stds = 3 # exclude values that are more than three standard deviations as in the book
    bins = 101 # book uses 100 bins
    
    changes_in_range = daily_relative_changes[(daily_relative_changes-mean)**2 <= (stds*std)**2]
    probability_density_array = pandas.cut(changes_in_range, bins=bins).value_counts(normalize=True, sort=False)
    
    r_min = probability_density_array.index.min().left
    r_max = probability_density_array.index.max().right
    
    # range of values that the next day's closing price can lie within for each bin
    x = numpy.arange(bins)/(bins-1) * (r_max - r_min) + r_min
    
    # method to calculate the probability that the next day's closing price will be r
    # I use a different method than the one in the book
    def p(r):
        
        binu = lambda off : min(max(0,int((bins-1) * (min(max(r,r_min),r_max)-r_min) / (r_max-r_min))+off),bins-1)
        pda = probability_density_array.iloc
        
        return 0.25 * (pda[binu(-1)] + pda[binu(1)] + 2*pda[binu(0)])
    
    # probability of the next day's closing price to be within a specific range
    y = numpy.vectorize(p)(x)
    
    # formula for the constant k as presented in the book
    k = lambda n : ((1.1924 + 33.2383*n + 56.2169*n*n) / (1 + 43.6196*n))**(1/3)
    
    dx = 2*stds*std/bins # range of each bin
    
    E0 = ((p(-dx) + p(dx)) / p(0) - 2) / dx / dx # ground state quantum price level
    
    l = (E0**3 - E0) / 1.1924 # lambda
    
    def E(n):
        
        nn = n if n >= 0 else -n
        
        c0 = -k(nn)**3*l
        c1 = (c0**2/4-1/27)**0.5
        
        E = (2*nn+1)*((c1-c0/2)**(1/3)-(c1+c0/2)**(1/3)) / E0
        EE = E if n >= 0 else -E
        
        return 1 + std * EE
    
    qpls = numpy.vectorize(E)(numpy.arange(61)-30) * daily_closing_prices.iloc[-1] # quantum price levels
    
    return x, y, qpls

@app.callback(
    Output(candle_chart_1m, 'figure'),
    Output(candle_chart_5m, 'figure'),
    Output(candle_chart_1d, 'figure'),
    Output(g_dist_plot, 'figure'),
    
    Input(update_int, 'n_intervals'), # n_intervals
    Input(tick_select, 'value'), # ticker
)

def update_figure(n_intervals, ticker):
    
    now = pandas.Timestamp.now('America/New_York')
    today = now.date()
    
    chart_lookback = 50
    
    try:
        daily_stock_data = yfinance.download(ticker.upper())
        minute_stock_data = yfinance.download(ticker.upper(), interval='1m', start=today, prepost=True)
        five_minute_stock_data = yfinance.download(ticker.upper(), interval='5m', start=today, prepost=True)
        x, y, qpls = calc_qpls(daily_stock_data['Adj Close'])
        
        daily_fig = create_candle_chart(daily_stock_data.iloc[-chart_lookback:], 'Daily OHLCV')
        min_fig = create_candle_chart(minute_stock_data.iloc[-chart_lookback:], 'Minute OHLCV', qpls, 3)
        five_min_fig = create_candle_chart(five_minute_stock_data.iloc[-chart_lookback:],
                                           'Five-Minute OHLCV', qpls, 5)
        
        gauss_fig = go.Figure(go.Bar(x=x, y=y, showlegend=False))
        
        gauss_fig.update(layout_xaxis_rangeslider_visible=False)
        gauss_fig.update_layout(title='Distribution of Relative Daily Price Changes',
                                yaxis_title='Probability', width=width, height=height)
        
        return min_fig, five_min_fig, daily_fig, gauss_fig
    
    except (ValueError, KeyError, IndexError) as error:
        print(error)
        
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
if __name__ == '__main__': app.run_server(debug=False)
