# Import required libraries
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
from TargetEncoder import TargetEncoder
from kNNModel import impute_model_kNN
import numpy as np
import plotly.graph_objects as go
from os.path import abspath, dirname

DATASET_PATH = dirname(abspath(__file__))


def clean_data (df):

	df = df.copy()
	df.loc[:,'time_elapsed'] = df.loc[:,'task_end_timestamp'].subtract(df.loc[:,'task_start_timestamp'])
	df['time_elapsed'] = df['time_elapsed'].replace(0, np.nan)
	part_column = df.part
	te = TargetEncoder('part')
	df = te.fit_transform(df, df['time_elapsed'])
	df['part_original'] = part_column
	df = impute_model_kNN(df, 5)
	return df

dft = pd.read_excel(DATASET_PATH + "/tasks.xls")
df = clean_data(dft)
num_devices = len(pd.unique(dft["part_id"]))


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server


# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                         html.P(
                            "Candidate: Mohammed CHAHBAR",
                            className="control_label",
                        
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Progression & Errors of Computer Tasks",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Overview", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                )
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Filter by Device IDs (or select range in histogram):",
                            className="control_label",
                        ),
                        dcc.RangeSlider(
                            id="id_slider",
                            min=1,
                            max=num_devices,
                            value=[1, num_devices],
                            className="dcc_control",
                        ),
                        html.P("Filter task/script by progression:", className="control_label"),
					    dcc.RadioItems(
					        id="progression_status",
					        options=[
					            {"label": "Tasks/Scripts with 100% progression", "value": "success"},
					            {"label": "Tasks/Scripts with with 0% Progression ", "value": "failed"},
					            {"label": "Tasks/Scripts with partial progression", "value": "middle"},
					            {"label": "All Tasks", "value": "all"},
						    ],
							value="success",
							labelStyle={"display": "inline-block"},
							className="dcc_control",
						),
						html.P("Filter tasks/script by error :", className="control_label"),
					    dcc.RadioItems(
					        id="error_status",
					        options=[
					            {"label": "Tasks/Scripts with 0 errors", "value": "zero"},
					            {"label": "Tasks/Scripts with errors ", "value": "errors"},
					            {"label": "all Tasks", "value": "all"},
					            {"label": "None", "value": "none"},
						    ],
							value="zero",
							labelStyle={"display": "inline-block"},
							className="dcc_control",
						),
                        
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="devNum_text"), html.P("Selected Number Of devices (See also the bar color change in unselected devices)")],
                                    id="wells",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="cnt_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


def filter_dframe(df, progStatus, id_slider, error_status):
	dataframes = []
	if progStatus == "success":
		progStatus = [100]
	elif progStatus == "failed":
		progStatus = [0]
	elif progStatus == "middle":
		progStatus = [i for i in range (1,100)]
	elif progStatus == "all":
		progStatus = [i for i in range (101)]
		print(progStatus)

	dff = df[df["percent"].isin(progStatus) ]
	dataframes.append(dff)

	if error_status != 'none':
		if error_status == 'zero':
			dff1 = df[df["num_of_errors"] == 0]
			dataframes.append(dff1)

		elif error_status == "errors":
			dff1 = df[df["num_of_errors"] != 0 	]
			dataframes.append(dff1)

		elif error_status == 'all':
			dff1 = df
			dataframes.append(dff1)
			
	return dataframes

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


# Selectors -> count graph
@app.callback(
    [Output("cnt_graph", "figure"),
    Output("id_slider", "max"),
    Output("devNum_text", "children")],
    [
        Input("progression_status", "value"),
        Input("id_slider", "value"),
        Input("error_status", "value"),
    ],
)
def make_figure(progStatus, id_slider, error_status):
	layout_count = copy.deepcopy(layout)
	dff_list = filter_dframe(dft, progStatus, [1, num_devices], error_status)
	colors_list =[{'selectColor':'rgb(228,171,149)', 'unselectColor': 'rgb(255,198,175)', 'colors':[]},
	{'selectColor':'rgb(205,92,92)', 'unselectColor': 'rgb(229,172,172)', 'colors':[]}]
	titles = ["Number of progressions per device", "number of tasks per device with/without errors"]
	indexs = [[] for i in range(len(dff_list))]
	curr_num_of_devices = 0
	fig = go.Figure()

	for i in range(len(dff_list)):
		tasks = dff_list[i].groupby(['part_id'])['task_id'].count()
		for j in range(1, num_devices + 1):
		    if j >= int(id_slider[0]) and j <= int(id_slider[1]):
		        colors_list[i]['colors'].append(colors_list[i]['selectColor'])
		        curr_num_of_devices+=1
		    else:
		    	colors_list[i]['colors'].append(colors_list[i]['unselectColor'])
		#create x__axis labels
		for j in tasks.index:
			indexs[i].append('device' + str(j))

		fig.add_trace(go.Bar(name=titles[i],
		x=indexs[i], 
		y=tasks, 
		marker_color=colors_list[i]['colors']))

	# Change the bar mode
	fig.update_layout(barmode='group',
	 title= "task progression & errors", 
	 dragmode= "select", 
	 showlegend= True, 
	 autosize= True,
	 legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))

    #to sunchronize slider with all the possible status
	if len(indexs) == 1:
		max_num_of_devices = len(indexs[0])
	else:
		max_num_of_devices = max([len(indexs[0]), len(indexs[1])])

	return fig, max_num_of_devices, curr_num_of_devices

# Slider -> count graph
@app.callback(Output("id_slider", "value"), [Input("cnt_graph", "selectedData")])
def update_id_slider(count_graph_selected):
	if count_graph_selected is None:
		return [1, num_devices]

	nums = [int(point["pointNumber"]) for point in count_graph_selected["points"]]
	return [min(nums) + 1, max(nums) + 1 + 1]
	


	
# Main
if __name__ == "__main__":
	app.run_server(debug=False, host='0.0.0.0', port = 8080)

