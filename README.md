# Interactive dashboard for computer tasks

This is a demo of the Dash interactive Python framework developed by [Plotly](https://plot.ly/).

Dash abstracts away all of the technologies and protocols required to build an interactive web-based application and is a simple and effective way to bind a user interface around your Python code. To learn more check out our [documentation](https://plot.ly/dash).

## Getting Started

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.

```
virtualenv venv

# Windows
venv\Scripts\activate
# Or Linux
source venv/bin/activate

```

Clone the git repo, then install the requirements with pip

```

git clone https://github.com/77experience/Interactive-dashboard
cd Interactive-dashboard
pip install -r requirements.txt

```

Run the app

```

python app.py

```

## About the app

This web app provide an interactive dashboard that visualize the computer tasks data from the tasks.xls dataset . Traces related to task progression and errors are ploted all together in the same graph as group so that we can obtain insights about the different variation of error and progression plots. Note also that we can display only the progression trace by setting errors to None in the filters.

On the graph we can select devices using the drag mode and also zoom on the graph. When selecting devices on the graph, notice the change in slider as well in the device counter above the graph. All this interactivity comes from the use of the so called "callbacks" in plotly.

## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots

