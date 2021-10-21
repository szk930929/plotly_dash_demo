# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output


app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

markdown_text = '''
### Dash and Markdown
'''

dfScatter = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

available_continent = dfScatter['continent'].unique()
opts = [{'label': 'all', 'value': 'all'}]
for i in available_continent:
    opts.append({'label': i, 'value': i})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
print(df)
available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([
  dcc.Markdown(children=markdown_text),
  html.Div([
      "Input: ",
      dcc.Input(id='my-input', value='initial', type='text')
  ]),
  html.Br(),
  html.Div(id='my-output'),
  dcc.Dropdown(
      id='continent-column',
      options=opts,
      value='all'
  ),
  dcc.Graph(
      id='life-exp-vs-gdp',
      # figure=fig
  ),
  html.Div([
      html.Pre(id='click-data', style=styles['pre']),
  ], className='three columns'),
  html.Div([
      html.Div([
          dcc.Dropdown(
              id='crossfilter-xaxis-column',
              options=[{'label': i, 'value': i} for i in available_indicators],
              value='Fertility rate, total (births per woman)'
          ),
          dcc.RadioItems(
              id='crossfilter-xaxis-type',
              options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
              value='Linear',
              labelStyle={'display': 'inline-block', 'marginTop': '5px'}
          )
      ], style={'width': '49%', 'display': 'inline-block'}),

      html.Div([
          dcc.Dropdown(
              id='crossfilter-yaxis-column',
              options=[{'label': i, 'value': i} for i in available_indicators],
              value='Life expectancy at birth, total (years)'
          ),
          dcc.RadioItems(
              id='crossfilter-yaxis-type',
              options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
              value='Linear',
              labelStyle={'display': 'inline-block', 'marginTop': '5px'}
          )
      ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
  }),
  html.Div([
      dcc.Graph(
          id='crossfilter-indicator-scatter',
          hoverData={'points': [{'customdata': 'Japan'}]}
      )
  ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

  html.Div([
      dcc.Graph(id='x-time-series'),
      dcc.Graph(id='y-time-series'),
  ], style={'display': 'inline-block', 'width': '49%'}),
  html.Div(dcc.Slider(
      id='crossfilter-year--slider',
      min=df['Year'].min(),
      max=df['Year'].max(),
      value=df['Year'].max(),
      marks={str(year): str(year) for year in df['Year'].unique()},
      step=None
  ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return '输出: {}'.format(input_value)


@app.callback(
    Output('life-exp-vs-gdp', 'figure'),
    Input('continent-column', 'value'))
def update_figure(selected_continent):
    if selected_continent == 'all':
        filtered_df = dfScatter
    else:
        filtered_df = dfScatter[dfScatter.continent == selected_continent]

    fig = px.scatter(filtered_df, x="gdp per capita", y="life expectancy",
                     size="population", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(clickmode='event+select')

    return fig


@app.callback(
    Output('click-data', 'children'),
    Input('life-exp-vs-gdp', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData)


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
          )

    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(dff, axis_type, title):

    fig = px.scatter(dff, x='Year', y='Value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True)
