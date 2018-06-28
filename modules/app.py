import sys
sys.path.insert(0, '../chem_scripts') # add path to chem_scripts

import dash
import os
import csv
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import base64

from CalcDescriptorUser import compute_descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from prototype_user2 import f_nn
from sklearn.preprocessing import MinMaxScaler

# add image location
imagefile = '../Poster/confusion_matrix.png'
encoded_image = base64.b64encode(open(imagefile, 'rb').read())

app = dash.Dash()

app.config['suppress_callback_exceptions']=True

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.layout = html.Div(children=[
	html.H1(children='ToxNet', style={'textAlign': 'center'}),
	html.H2(children='Prediction Models for Chemical Toxicity.', style={'textAlign': 'center'}),
	html.Hr(),
	html.Br(),
	html.Br(),
	html.Div([
		html.Label('SMILE Structure ( i.e. CCN(CC)CCCl ) '),
		dcc.Input(
			id='smile',
			placeholder='Enter a string...',
			type='text',
			value=''
		),

		html.Br(),
		html.Br(),
		html.Label('Select Task'),
		dcc.Dropdown(
			id='dropdown_task',
			options=[
				{'label': 'nontoxic', 'value': 'nontoxic'},
				{'label': 'verytoxic', 'value': 'verytoxic'},
				{'label': 'GHS', 'value': 'ghs'},
				{'label': 'EPA', 'value': 'epa'},
				{'label': 'LD50', 'value': 'logld50'}
			],
			value='',
		),

		html.Br(),
		html.Label('Select Networks'),
		dcc.Dropdown(
			id='dropdown_net',
			options=[
				{'label': 'MLP', 'value': 'mlp'},
				#{'label': 'Recurrent Neural Network', 'value': 'rnn'},
				#{'label': 'Convolutional Neural Network', 'value': 'cnn'},
			],
			value=''
		),

		html.Br(),
		html.Br(),
		# submit bottom
		html.Button(id='button', n_clicks=0, children='Submit'),
		html.Br(),

		html.Hr(),
		html.Br(),
		html.H4(
			'''
			Part 1. Figures for training process
			'''),
		html.Br(),
		html.Br(),
		html.Div([
			dcc.Graph(id='loss curve'),
			html.Img(
			src='data:image/png;base64,{}'.format(encoded_image.decode()), 
			style={
				'width': '450px',
				'height': '450px',
		    	'float': 'center',
			})
		], style={'columnCount': 2}),

		html.Br(),
		html.H4(
			'''
			Part 2. Overall Prediction
			'''
		),

		html.Div(
			className="row",
			children=[
				html.Div(
					className="six columns",
					children=[
						html.Div(
							children=dcc.Graph(
								id='acc_bar',
								figure={
									'data': [
										{'x': ['nontoxic', 'verytoxic', 'GHS', 'EPA'],
										 'y': [0.724, 0.788, 0.751, 0.824],
										 'type': 'bar',
										 'name': 'CNN'},
										{'x': ['nontoxic', 'verytoxic', 'GHS', 'EPA'],
										 'y': [0.792, 0.859, 0.814, 0.842],
										 'type': 'bar',
										 'name': 'RNN'},
										{'x': ['nontoxic', 'verytoxic', 'GHS', 'EPA'],
										 'y': [0.838, 0.859, 0.814, 0.842],
										 'type': 'bar',
										 'name': 'MLP'}
									],
									'layout': {
										'title': 'Overall Test Result',
										'width': 600,
										'height': 500,
										'margin': {'l':50, 'b':80, 't':80, 'r':0},
										'xaxis': {'title':'Task'},
										'yaxis': {'title':'AUC value'},
									}
								}
							)
						)
					]
				),
				html.Div(
					[html.H4(children='Part 3. Table of Possibility'), 
					 html.Table(id='predict_table')],
					 style={'width': '100%','display': 'inline-block', 'padding': '0 10', 'float': 'center'},
				)
			]
		)
	])
])


@app.callback(
	dash.dependencies.Output('loss curve', 'figure'),
	[dash.dependencies.Input('button', 'n_clicks')],
	[dash.dependencies.State('dropdown_task', 'value'),
	dash.dependencies.State('dropdown_net', 'value')])

def loss_result(n_clicks, task, network):
	if n_clicks != 0:
		df_loss = pd.read_csv('../result/'+network+'/tox_niehs_'+network+'_'+task+'_loss_curve_1_0.csv')
		traces=[]
		for i in ['loss', 'val_loss']:
			traces.append(go.Scatter(
				x=df_loss['epoch'],
				y=df_loss[i],
				name=i,
				mode='lines+markers',
				text=i,
				marker={
					'size': 15,
					'opacity': 0.5,
					'line': {'width': 0.5, 'color': 'white'}
				})
			)
		return {
			'data':traces,
			'layout': go.Layout(
				title='Loss Curve',
			    autosize=False,
				width=600,
				height=500,
			    margin=go.Margin(l=0, r=50, b=100, t=100, pad=4
			    ),
				xaxis={'title': 'epoch'},
				yaxis={'title': 'loss value'},
				showlegend=True,
				hovermode='closest'
				)
		}
	else :
		return 

@app.callback(
	dash.dependencies.Output('predict_table', 'children'),
	[dash.dependencies.Input('button', 'n_clicks')],
	[dash.dependencies.State('smile', 'value'),
	dash.dependencies.State('dropdown_net', 'value'),
	dash.dependencies.State('dropdown_task', 'value')])
def predict_result(n_clicks, smile, network, task):
	if n_clicks != 0:
		if input == None:
			raise Exception('Please Enter Smile Structure!')
		if network == 'mlp':
			# generate descriptors for MLP
			mol = Chem.MolFromSmiles(smile)
			descriptors = compute_descriptors(mol, 'molid 0')

			df_new = []
			df_new.append(descriptors)
			all_new = np.asarray(df_new)
			all_desc = all_new[:,1:].astype(float)
			all_name = all_new[:,:1]

			all_desc = all_desc[:,~np.any(np.isnan(all_desc), axis=0)]
			
			if len(all_desc[0]) == 0:
				return "Please input another string! This string is broken."
			else:
				namelist = np.arange(all_desc.shape[1]).tolist()
				namelist.insert(0, 'id')
				all_combined = np.concatenate((all_name, all_desc), axis=1)
				df_test = pd.DataFrame(np.asarray(all_combined), columns=namelist)

				df_test = df_test.drop('id', axis=1)
				# drop columns (source from merge descriptors + 1)
				df_test= df_test.drop([16, 17, 18, 19], axis=1)

				''' For Test ONLY!!!
				df = pd.read_csv("../data/tox_niehs_int_"+task+"_rdkit.csv").drop(columns=[task, 'id'])
				df_test = pd.DataFrame()
				df_test = df_test.append(df.iloc[50,:], sort=False)
				'''

				y_pred = f_nn(task, network, df_test)
				df_result = pd.DataFrame(y_pred, columns=['class '+str(i) for i in np.arange(np.size(y_pred, 1))])
				return generate_table(df_result)
				'''
					####### RNN can only be used in GPU and with cudnn registered ########
				'''
				
		else:
			return 
	else:
		return



app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__=='__main__':
	app.run_server(debug=True)