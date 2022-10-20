# Run this app with `python main.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from src.dashboard import prediction_plots, explainability_plots


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__, title="Wind Power Forecasting", external_stylesheets=external_stylesheets
)
app.config.suppress_callback_exceptions = True
server = app.server


def prediction_tab(figure, rmse, r2, maape):
    layout = html.Div(
        [
            html.H3("Prediction Graph"),
            html.Div(children="""Prediction for 01-12-2018 to 31-12-2018."""),
            dcc.Graph(
                id="comparison-graph",
                figure=figure,
            ),
            html.Div(
                dcc.Markdown(
                    f"""
                    #### Test set evaluation
                    **RMSE**: {rmse}  
                    **R2 Score**: {r2}  
                    **MAAPE**: {maape}
                    """
                )
            ),
        ]
    )
    return layout


def explainability_tab(fig1, fig2):
    layout = html.Div(
        [
            html.H3("Feature Importances and Wind Rose Chart"),
            html.Div(
                children=[
                    dcc.Graph(
                        id="feature-importances",
                        figure=fig1,
                        style={"display": "inline-block"},
                    ),
                    dcc.Graph(
                        id="windrose",
                        figure=fig2,
                        style={"display": "inline-block"},
                    ),
                ]
            ),
            html.Div(
                dcc.Markdown(
                    """
                    Feature engineering was done as follows:  
                    1. Wind direction is binned into 4 regions:  
                    &emsp; 1. North - East  
                    &emsp; 2. East - South  
                    &emsp; 3. South - West  
                    &emsp; 4. West - North  
                    2. These regions are one-hot encoded into 4 categorical columns.  
                    3. `Gust` feature is the difference between each timestamp of wind speed.  
                    4. MinMaxScaler was applied to each of the numerical columns.
                    """
                )
            ),
        ]
    )
    return layout


app.layout = html.Div(
    [
        html.Div(
            children=[
                html.P(
                    children="💨",
                    className="header-emoji",
                ),
                html.H1("Wind Power Forecasting", className="header-title"),
                html.P(
                    children="Depicts how much wind power is to be expected at particular instant of time in the days to come.",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(children="Model", className="menu-title"),
                dcc.Dropdown(
                    options=["LinearRegression", "XGBoost", "LightGBM"],
                    value="LinearRegression",
                    clearable=False,
                    id="model-name",
                ),
                html.Div(children="Aggregation", className="menu-title"),
                dcc.Dropdown(
                    options=["Original (10 minutes)", "Hourly", "Daily"],
                    value="Original (10 minutes)",
                    clearable=False,
                    id="agg-type",
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                dcc.Tabs(
                    id="wind-graph",
                    value="prediction-graph",
                    children=[
                        dcc.Tab(label="Prediction", value="prediction-graph"),
                        dcc.Tab(
                            label="Model Explainability", value="explainability-graph"
                        ),
                    ],
                ),
                html.Div(id="wind-content-graph"),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    Output("wind-content-graph", "children"),
    Input("wind-graph", "value"),
    Input("model-name", "value"),
    Input("agg-type", "value"),
)
def render_content(tab, name, agg):
    compare_fig, rmse, r2, maape = prediction_plots(name, agg)
    feature_fig, windrose_fig = explainability_plots(name)
    if tab == "prediction-graph":
        return prediction_tab(compare_fig, rmse, r2, maape)
    elif tab == "explainability-graph":
        return explainability_tab(feature_fig, windrose_fig)


if __name__ == "__main__":
    import os

    debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
    app.run_server(host='0.0.0.0', port=8050, debug=debug)
