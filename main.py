

import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import data_read
import K_means
import Decision_tree
import linear_regression
import support_vector_machine
import chart_studio as cs
import fcm
import dash_bootstrap_components as dbc
from datetime import datetime

data =data_read.data_read()
load_df = pd.DataFrame(data, columns=['id', 'date', 'energy_use'])
loads_df = data_read.data_process()
unique_days = loads_df.day_of_month.unique()
localtime = datetime.today()
# external_stylesheets = ['/css/my.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.config["suppress_callback_exceptions"] = True

# COLORES
colors = {
    'background': '#f7f7f7',
    'text': '#00a99e',
    'title': '#99b9b7',
    'subtitle': '#99b9b7',
    'Graph': '#d4d4d4',
    'text1': '#080909',
    'GraphLine': '#f4d44d',
    'Alto': '#99e699',
    'Medio': '#aae2fb',
    'Bajo': '#ff9999'
}

cards = [
    dbc.Card(
        [
            html.H2(f"{load_df['id'].value_counts().shape[0]}           ", className="card-title"),
            html.P("用户数", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{load_df.shape[0]}           ", className="card-title"),
            html.P("数据量", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{round(float(load_df['energy_use'].max()),2)}KW", className="card-title"),
            html.P("峰值", className="card-text"),
        ],
        body=True,
        color= colors['text'],
        inverse=True,
    ),
]

def elaboracion_encabezado():
    return [
                html.Div(
                [
            html.H1(children='用电控制系统数据分析端',
                    style={'margin-left': '20px','margin-top': '50px','textAlign': 'left','color': colors['text'],'size': '100px'},
                     className="nine_columns"),
            html.Img(src= app.get_asset_url('logo1.png'),
                     className="three columns",
                     style={'height': '15%','width': '15%','float': 'right','position':'relative'}),
                 ],className="row1"),
        ]
# def elaboracion_encabezado():
#     return [
#         html.Div([
#             html.Div([
#                 html.Img(src=app.get_asset_url('logo1.png'), height='15%', width='150')
#             ], className="ten columns padded")
#         ], className="row gs-header"),
#
#         html.Div([
#             html.Div([
#                 html.H1(
#                 '多场景物联网用电控制系统之数据分析端')
#            ], className="twelve columns padded")
#        ], className="row gs-header gs-text-header")
#
#     ]
    # return [
    #     html.Div(
    #         [
    #             html.H1(children='用电控制系统之数据分析端',
    #                     style={'margin-left': '20px', 'margin-top': '50px', 'textAlign': 'left',
    #                            'color': colors['text']},
    #                     className="row"),
    #             html.Img(src=app.get_asset_url('logo1.png'),
    #                      className="three columns",
    #                      style={'height': '15%', 'width': '15%', 'float': 'right', 'position': 'relative'}),
    #         ], className="row"),
    # ]


def row1():
    return [
        html.Div([
            html.Div([
                html.Br([]),
                html.H6('数据背景',
                        className="gs-header gs-text-header padded"),

                html.Br([]),
                dcc.Markdown('''   数据获取于基于蓝牙BLE的多场景物联网用电控制系统的数据服务端，
               本次模拟采用的数据源自Pecan Street Energy Database的数据库文件dataport_sqlite'''),
            ], className="row")
        ], className="row ")
    ]

def users_classical():
    return [
        html.Div([
            html.Div([
                html.Br([]),
                html.H6('用户分类',
                        className="gs-header gs-text-header padded"),

                html.Br([]),
                dcc.Markdown('''  用户分类功能描述 '''),
            ], className="row")
        ], className="row ")
    ]
def load_forecast():
    return [
        html.Div([
            html.Div([
                html.Br([]),
                html.H6('负荷预测',
                        className="gs-header gs-text-header padded"),

                html.Br([]),
                dcc.Markdown('''  负荷预测功能描述 '''),
            ], className="row")
        ], className="row ")
    ]
def row2():
    return html.Div([

        html.H6('用电用户数据情况',
                className="gs-header gs-text-header padded"),
        html.Br([]),

        html.Div([
            html.Div([
                html.H6(children='选择用户：',
                    style={'textAlign': 'left', 'color': colors['text'],'size':'20px'}
            )],className='four_col'),
            html.Div([
                dcc.Dropdown(
                    id='user_id',
                    options=[{'label': i, 'value': i}
                            for i in sorted(loads_df['id'].unique())],
                    value=26
                )], className="four_col"),
        ],className='flex_start'),
        html.Div([
            dcc.Graph(id='user_data'),
        ], className="row"),
    ],className='row')
def row3():
    return html.Div([
        html.Div([
            html.Div([
                html.H6(["K-means用户分类"],
                        className="gs-header gs-table-header padded"),
            ]),
            html.Div([
                html.Div([
                    html.H6(children='选择日期：',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='four_col'),
                html.Div([
                    dcc.Dropdown(
                        id='day1',
                        options=[{'label': i, 'value': i}
                                 for i in unique_days],
                        value=1
                    )], className='four_col'),
                html.Div([
                    html.H6(children='号',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='col'),
            ],className='flex_start'),

            html.Div([
                dcc.Graph(id='kmeans_user_classical'),
            ], className="row")
        ], className="five columns"),

        html.Div([
            html.H6(["FCM用户分类"],
                className="gs-header gs-table-header padded"),

            html.Div([

                html.Div([
                    html.H6(children='选择日期：',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='four_col'),
                html.Div([
                    dcc.Dropdown(
                        id='day2',
                        options=[{'label': i, 'value': i}
                                 for i in unique_days],
                        value=1
                    )], className='four_col'),
                html.Div([
                    html.H6(children='号',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='col'),
            ],className='flex_start'),
            html.Div([
                dcc.Graph(id='fcm_user_classical'),
            ], className="row"),

        ], className="seven columns"),
    ],className='flex')

@app.callback(
    Output("kmeans_user_classical", "figure"),
    [Input("day1", "value")],
)
def update_figure1(selected_day):
    loads_wide_df = K_means.all_user_process(selected_day)
    load_data = np.array(loads_wide_df)
    predictions = K_means.k_means_train(load_data)
    means = []
    for i in range(0, 4):
        all_data =[]

        for x,y in zip(load_data,predictions):
            if y == i:
                all_data.append(x)
        all_data_array = np.array(all_data)
        mean = all_data_array.mean(axis=0)
        means.append(mean)


    trace1 = go.Scatter(
        # x=[i+1 for i in range(len(means[0]))],
        x=loads_wide_df.T.index,
        y=means[0],
        mode='lines',
        name='第一类用户'
    )
    trace2 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[1],
        mode='lines',
        name='第二类用户'
    )
    trace3 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[2],
        mode='lines',
        name='第三类用户'
    )
    trace4 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[3],
        mode='lines',
        name='第四类用户'
    )
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis=dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output("fcm_user_classical", "figure"),
    [Input("day2", "value")],
)
def update_figure2(selected_day):
    loads_wide_df = fcm.all_user_process(selected_day)
    load_data = np.array(loads_wide_df)
    predictions = fcm.fcm_train(load_data)
    means = []
    for i in range(0, 4):
        all_data =[]

        for x,y in zip(load_data,predictions):
            if y == i:
                all_data.append(x)
        all_data_array = np.array(all_data)
        mean = all_data_array.mean(axis=0)
        means.append(mean)


    trace1 = go.Scatter(
        # x=[i+1 for i in range(len(means[0]))],
        x=loads_wide_df.T.index,
        y=means[0],
        mode='lines',
        name='第一类用户'
    )
    trace2 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[1],
        mode='lines',
        name='第二类用户'
    )
    trace3 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[2],
        mode='lines',
        name='第三类用户'
    )
    trace4 = go.Scatter(
        # x=[i+1 for i in range(len(means[1]))],
        x=loads_wide_df.T.index,
        y=means[3],
        mode='lines',
        name='第四类用户'
    )
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis=dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def all_user_data_graph():
    return [
        html.Div(id='TMPM', children=[
            html.Div([

                dcc.Dropdown(
                    id='user_id',
                    options=[{'label': i, 'value': i}
                                for i in sorted(loads_df['id'].unique())],
                    value=26
                )], className="six columns"),
            html.Div([
                dcc.Graph(id='user_data'),
            ], className="row"),


        ]),
    ]

@app.callback(
        Output("user_data", "figure"),
        [Input("user_id", "value")],
    )
def update_figure0(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = linear_regression.linear_train(X, Y)
    trace1 = go.Scatter(
        x=filtered_df[filtered_df['id'] == selected_user_id]['date'],
        y=filtered_df[filtered_df['id'] == selected_user_id]['energy_use'],
        mode='lines',
        name='真实值'
    )
    data = [trace1]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis=dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def row4():
    return html.Div([
        html.Div([
            html.H6(["决策树回归负荷预测"],
                className="gs-header gs-table-header padded"),
            html.Div([
                html.Div([
                html.H6(children='选择用户：',
                        style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                        )], className='four_col'),
                html.Div([
                    dcc.Dropdown(
                        id='user_id1',
                        options=[{'label': i, 'value': i}
                                 for i in sorted(loads_df['id'].unique())],
                        value=26
                    )
                ],className="four_col")
                ], className="flex_start"),
            html.Div([
                dcc.Graph(id='decision_tree'),
            ], className="row"),
            html.H6(id="tree_evaluate", style={'color': colors['text'], 'size': '30px'}, className="row")
            # dbc.Label(id="tree_evaluate", style={'color': colors['text'], 'size': '30px'}, className="row")
        ], className="five columns"),


        html.Div([
            html.H6(["线性回归负荷预测"],
                className="gs-header gs-table-header padded"),
            html.Div([
                html.Div([
                    html.H6(children='选择用户：',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='four_col'),
                html.Div([
                    dcc.Dropdown(
                        id='user_id2',
                        options=[{'label': i, 'value': i}
                                 for i in sorted(loads_df['id'].unique())],
                        value=26
                    )
                ], className="four_col")
            ], className="flex_start"),
            html.Div([
                dcc.Graph(id='linear_regression'),
            ], className="row"),
            # dbc.Label(id="linear_evaluate", style={'color': colors['text'], 'size': '30px'},className="row")
            html.H6(id="linear_evaluate", style={'color': colors['text'], 'size': '30px'}, className="row")
        ], className="seven columns"),
    ],className='flex')
@app.callback(
    Output("decision_tree", "figure"),
    [Input("user_id1", "value")],
)
def update_figure3(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = Decision_tree.tree_train(X, Y)

    trace1 = go.Scatter(
        x=filtered_df['date'][-100:-1],
        y=filtered_df['energy_use'][-100:-1],
        mode='lines',
        name='真实值'
    )
    trace2 = go.Scatter(
        x=filtered_df['date'][-100:],
        y=pre[-100:],
        mode='lines+markers',
        name='预测值'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis=dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output("tree_evaluate", "children"),
    [Input("user_id1", "value")],
)
def update_tree_evaluate(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = Decision_tree.tree_train(X, Y)
    r2, result= Decision_tree.evaluate(Y,pre)
    return f"算法r方值：{round(r2,2)}  涨跌预测率 ：{result}%"

@app.callback(
    Output("linear_regression", "figure"),
    [Input("user_id2", "value")],
)
def update_figure4(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = linear_regression.linear_train(X, Y)
    trace1 = go.Scatter(
        x=filtered_df['date'][-100:-1],
        y=filtered_df['energy_use'][-100:-1],
        mode='lines',
        name='真实值'
    )
    trace2 = go.Scatter(
        x=filtered_df['date'][-100:],
        y=pre[-100:],
        mode='lines+markers',
        name='预测值'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis = dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output("linear_evaluate", "children"),
    [Input("user_id2", "value")],
)
def update_linear_evaluate(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = Decision_tree.tree_train(X, Y)
    r2, result= linear_regression.evaluate(Y,pre)
    return f"算法r方值：{round(r2,2)}  涨跌预测率 ：{result}%"

def row5():
    return html.Div([
        html.Div([
            html.H6(["支持向量机回归负荷预测"],
                    className="gs-header gs-table-header padded"),
            html.Div([
                html.Div([
                    html.H6(children='选择用户：',
                            style={'textAlign': 'left', 'color': colors['text'], 'size': '20px'}
                            )], className='four_col'),
                html.Div([
                    dcc.Dropdown(
                        id='user_id3',
                        options=[{'label': i, 'value': i}
                                 for i in sorted(loads_df['id'].unique())],
                        value=26
                    )
                ], className="four_col")
            ], className="flex_start"),
            html.Div([
                dcc.Graph(id='svm'),
            ], className="row"),
            # dbc.Label(id="svm_evaluate", className="row")
            html.H6(id="svm_evaluate", style={'color': colors['text'], 'size': '30px'}, className="row")
        ], className="five columns"),
        html.Div([
            html.H6(["LSTM负荷预测"],
                    className="gs-header gs-table-header padded"),

        ], className="seven columns"),
    ], className='flex')

@app.callback(
    Output("svm", "figure"),
    [Input("user_id3", "value")],
)
def update_figure5(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = support_vector_machine.svm_train(X, Y)

    trace1 = go.Scatter(
        x=filtered_df['date'][-100:-1],
        y=filtered_df['energy_use'][-100:-1],
        mode='lines',
        name='真实值'
    )
    trace2 = go.Scatter(
        x=filtered_df['date'][-100:],
        y=pre[-100:],
        mode='lines+markers',
        name='预测值'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        xaxis=dict(title="日期"),
        yaxis=dict(title="kw"),
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output("svm_evaluate", "children"),
    [Input("user_id3", "value")],
)
def update_svm_evaluate(selected_user_id):
    filtered_df = loads_df[loads_df.id == selected_user_id]
    X, Y = data_read.x_y_data_process(filtered_df)
    pre = support_vector_machine.svm_train(X, Y)
    r2, result= support_vector_machine.evaluate(Y,pre)
    return f"算法r方值：{round(r2,2)}  涨跌预测率 ：{result}%"

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div([
        # ENCABEZADO
        html.Div(elaboracion_encabezado()),
        html.Div(row1()),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Div(row2()),
        html.Div(users_classical()),
        html.Div(row3()),
        html.Div(load_forecast()),
        html.Div(row4()),
        html.Div(row5()),
        # html.Div(all_user_data_graph()),

        html.Div(id='secciones', children=[
            # CONTENIDO
            html.Div(id="app-content"),

        ])
    ], className="subpage")
])

if __name__ == '__main__':
    app.run_server(debug=True)


@app.callback(
    Output('historic-graph', 'figure'),
    [Input('day-slider', 'value')]
)
def update_figure(selected_date):
    return






























