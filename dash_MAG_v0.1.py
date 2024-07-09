import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import os 
app = dash.Dash(__name__)

# Read the data from the CSV file
df = pd.read_csv('combined_metadata.csv')
df_raw = pd.read_csv('combined_metadata.csv')
df = df.fillna('NA')

# reformat classificaiton  for readibility 
def format_classification(text, break_after=3):
    # Detect the delimiter by checking the occurrence
    delimiter = ';' if ';' in text else '/'
    
    parts = text.split(delimiter)
    if len(parts) > break_after:
        # Insert a line break after the third part
        parts[break_after] = "<br>" + parts[break_after]
    return delimiter.join(parts)
    
df['formatted_classification'] = df['classification'].apply(format_classification)

def is_numeric(series):
    """ Check if a pandas series is numeric. """
    return pd.api.types.is_numeric_dtype(series)


# Define taxonomic ranks
tax_ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# Split the classification column into separate columns for each level
df[tax_ranks] = df['classification'].str.split(';', expand=True)

metadata_columns = [col for col in df.columns if col not in tax_ranks + ['classification', 'Genome']]

dropdown_options = [{'label': col, 'value': col} for col in metadata_columns if col != 'formatted_classification']

def create_sunburst(selected_metadata):
    path = tax_ranks + [selected_metadata]
    sunburst_fig = px.sunburst(
        df, 
        path=path,
        color=selected_metadata,
        
        color_continuous_scale=px.colors.sequential.RdBu,
        maxdepth=7  # This will show up to species level (6) plus one more for metadata
    )
    sunburst_fig.update_layout(height=700, width=700) 
    return sunburst_fig



section_style = {
    'backgroundColor': '#f0f0f0',
    'padding': '20px',
    'margin': '10px 0',
    'borderRadius': '5px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
}
common_style = {
    'border': '2px solid #ddd',
    'borderRadius': '5px',
    'padding': '20px',
    'marginBottom': '20px',
    'backgroundColor': 'white',
}
"""
    html.Div([
        html.Div([
            html.H2('Quality and number of all genomes', style={'textAlign': 'left'}),

            #html.H2('Quality and number of genomes', style={'textAlign': 'left', 'marginLeft': '20px'}),
            html.Label('Select your metadata', style={'fontSize': 15, 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='color-dropdown',
                options=[{'label': col, 'value': col} for col in metadata_columns],
                value=metadata_columns[0] if metadata_columns else None,
                placeholder="Select color variable",
                style={
                    'width': '300px', 
                    'height': '30px',
                    'fontSize': 16,
                    'whiteSpace': 'normal',
                    'textOverflow': 'ellipsis'
                }
            ),
        ], style={'width': '100%', 'margin': '10px auto'}),
        #**section_style,
    ], style={**section_style}),
    #], style={'backgroundColor': 'white', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='overall-scatter-plot', style={'height': '600px'})
    ], style={'width': '65%', 'display': 'inline-block','backgroundColor': '#f0f0f0'}),
    html.Div([
        dcc.Graph(id='overall-distribution-plot', style={'height':'600px'})
    ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top','backgroundColor': '#f0f0f0'}),
"""

app.layout = html.Div([
    html.H1('metaFun : genome selector for COMPARATIVE_ANNOTATION', style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H2('Quality and number of all genomes', style={'textAlign': 'left'}),
            html.Label('Select your metadata', style={'fontSize': 15, 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='color-dropdown',
                options=[{'label': col, 'value': col} for col in metadata_columns if col != 'formatted_classification'],
                value=metadata_columns[0] if metadata_columns else None,
                placeholder="Select color variable",
                style={
                    'width': '300px', 
                    'height': '30px',
                    'fontSize': 16,
                    'whiteSpace': 'normal',
                    'textOverflow': 'ellipsis'
                }
            ),
        ], style=section_style),#{'marginBottom': '20px'}),

        html.Div([
            dcc.Graph(id='overall-scatter-plot', style={'height': '600px'})
        ], style={'width': '65%', 'display': 'inline-block', 'overflow': 'visible'}),
        html.Div([
            dcc.Graph(id='overall-distribution-plot', style={'height':'600px'})
        ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top', 'overflow': 'visible'}),
    ], style={**common_style, 'overflow': 'visible'}),    

    html.Div([
        html.Div([
            html.H2('Select genomes that you are interested', style={'textAlign': 'left'}),

            html.Label('Select your metadata for subsetting genomes', style={'fontSize': 15, 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='metadata-dropdown',
                    options=dropdown_options,
                    value='pass.GUNC',  # Default value
                    style={'width': '300px'}
                ),
            ],style={ 'width':'100%','margin': '10px auto','textAlign': 'left'}),
        ], style=section_style),#style={'padding': '20px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='sunburst-plot')
            ],  style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '100%'}),
            #style={'width': '80%', 'display': 'inline-block','overflow': 'visible'}),
        ], style={'overflow': 'visible'}),

        html.Div([
            dcc.Graph(id='filtered-scatter-plot', style={'width': '65%', 'display': 'inline-block', 'height': '600px'}),
            dcc.Graph(id='filtered-distribution-plot', style={'width': '35%', 'display': 'inline-block', 'height': '600px'})
        ], style={'width': '100%', 'display': 'flex', 'overflow': 'visible'}),


        html.Div([
            dcc.Input(
                id='search-input',
                type='text',
                placeholder='Search table...',
                style={'marginRight': '10px', 'width': '200px'}
            ),
            dcc.Input(
                id='filename-input',
                type='text',
                placeholder='Enter filename for CSV',
                style={'marginRight': '10px', 'width': '200px'}
            ),
            html.Button('Save to Local', id='save-local-button', n_clicks=0),
        ], style={'marginBottom': '10px', 'marginTop': '20px'}),

        html.Div([
            dcc.Dropdown(
                id='column-selector',
                options=[{'label': col, 'value': col} for col in df.columns  if col != 'formatted_classification'],
                value=[col for col in df.columns[:10]],  # default selected columns
                multi=True,
                placeholder="Select columns to display"
            )
        ], style={'width': '50%', 'margin': '10px auto'}),

        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns[:10] if i != 'formatted_classification'],  # Adjust number of columns displayed        
            data=df.head(10).to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'scroll'},  # Enable horizontal scroll
            filter_action='custom',
            filter_query=''
        ),
        dcc.Download(id="download-dataframe-csv"),
        html.Div(id='save-status') 
    ], style=common_style)
])



@app.callback(
    [Output('overall-scatter-plot', 'figure'),
     Output('overall-distribution-plot', 'figure')],
    [Input('color-dropdown', 'value')]
)
def update_overall_plots(color_var):
    scatter_fig = px.scatter(
        df, 
        x='Completeness', 
        y='Contamination', 
        color=color_var,
        hover_data=['formatted_classification'],
        title='Overall: Completeness vs Contamination'
    )
    scatter_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    dist_fig = px.histogram(
        df, 
        x=color_var,
        title=f'Overall Distribution of {color_var}'
    )
    dist_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    return scatter_fig, dist_fig

@app.callback(
    Output('sunburst-plot', 'figure'),
    Input('metadata-dropdown', 'value')
)
def update_sunburst(selected_metadata):
    return create_sunburst(selected_metadata)

@app.callback(
    [Output('filtered-scatter-plot', 'figure'),
     Output('filtered-distribution-plot', 'figure')],
    [Input('sunburst-plot', 'clickData'),
     Input('metadata-dropdown', 'value')],
    [State('sunburst-plot', 'figure')]
)



def update_filtered_plots(clickData, selected_metadata, current_figure):
    filtered_df = df.copy()
    
    if clickData:
        current_path = clickData['points'][0]['id'].split('/')
        for i, level in enumerate(current_path):
            if level and i < len(tax_ranks):
                filtered_df = filtered_df[filtered_df[tax_ranks[i]] == level]
    
    if filtered_df.empty:
        return dash.no_update, dash.no_update
    
    hover_data = ['formatted_classification'] + metadata_columns

    scatter_fig = px.scatter(
        filtered_df, 
        x='Completeness', 
        y='Contamination', 
        color=selected_metadata,
        hover_data=hover_data,
        title='Filtered: Completeness vs Contamination'
    )
    
    scatter_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    metadata_dist_fig = px.histogram(
        filtered_df, 
        x=selected_metadata,
        title=f'Filtered Distribution of {selected_metadata}'
    )
    metadata_dist_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    return scatter_fig, metadata_dist_fig

@app.callback(
    Output('data-table', 'data'),
    [Input('search-input', 'value'),
     Input('sunburst-plot', 'clickData'),
     Input('metadata-dropdown', 'value')]
)
def update_table(search_value, clickData, selected_metadata):
    filtered_df = df.copy()
    
    if clickData:
        current_path = clickData['points'][0]['id'].split('/')
        for i, level in enumerate(current_path):
            if level and i < len(tax_ranks):
                filtered_df = filtered_df[filtered_df[tax_ranks[i]] == level]
    
    if search_value:
        filtered_df = filtered_df[filtered_df.apply(lambda row: any(str(search_value).lower() in str(cell).lower() for cell in row), axis=1)]
    return filtered_df.to_dict('records')

#    return filtered_df.head(10).to_dict('records')

@app.callback(
    Output('data-table', 'columns'),
    Input('column-selector', 'value')
)
def update_table_columns(selected_columns):
    valid_columns = [col for col in selected_columns if col != 'formatted_classification']
    return [{"name": col, "id": col} for col in valid_columns]
    #return [{"name": i, "id": i} for i in selected_columns if i != 'formatted_classification']

# download selected genome in the users local server callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("save-local-button", "n_clicks"),
    State("filename-input", "value"),
    State("data-table", "data"),
    prevent_initial_call=True,
)
def save_to_local(n_clicks, filename, table_data):
    if n_clicks == 0:
        raise PreventUpdate
    
    if not filename:
        filename = "data"

    filtered_df = pd.DataFrame(table_data)
#    matched_df = df[df.index.isin(filtered_df.index)]
    matched_df = df_raw[df_raw.index.isin(filtered_df.index)]

    return dcc.send_data_frame(matched_df.to_csv, f"{filename}.csv", index=False)

#    full_filtered_df = df.loc[filtered_df.index]

#    return dcc.send_data_frame(full_filtered_df.to_csv, f"{filename}.csv", index=False)

#    return dcc.send_data_frame(df.to_csv, f"{filename}.csv", index=False)

# 서버 저장 콜백
"""
@app.callback(
    Output("save-status", "children"),
    Input("save-server-button", "n_clicks"),
    State("filename-input", "value"),
    State("data-table", "data"),
    prevent_initial_call=True,
)
def save_to_server(n_clicks, filename, table_data):
    if n_clicks == 0:
        raise PreventUpdate
    
    if not filename:
        filename = "data"
    
    df = pd.DataFrame(table_data)
    server_path = os.path.join(os.getcwd(), f"{filename}.csv")
    df.to_csv(server_path, index=False)
    return f"File saved to server at {server_path}"
"""

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
    #app.run_server(debug=True)