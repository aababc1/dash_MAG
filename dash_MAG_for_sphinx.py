import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import os 
import numpy as np
import argparse 
import dash_daq as daq 

parser = argparse.ArgumentParser(description='metaFun: genome selector for COMPARATIVE_ANNOTATION')
parser = argparse.ArgumentParser(description='refer to the documentation at https://metafun-doc-v01.readthedocs.io/en/latest/')
parser.add_argument('-i', '--input', help='Input CSV file', required=True)
args = parser.parse_args()


app = dash.Dash(__name__)

# Read the data from the CSV file
df = pd.read_csv(args.input)
df_raw = pd.read_csv(args.input)


#df = df.fillna('NA')
#df = df.apply(lambda col: col.fillna(-999) if col.dtype.kind in 'biufc' else col.fillna('Missing'))
df = df.fillna('None')
"""
def prepare_data(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1)  # Or another appropriate marker
        else:
            df[col] = df[col].fillna('None')
    return df

df = prepare_data(df)
"""
def create_scatter_plot(df, x_col, y_col, color_col):
    # Mask to identify rows with placeholder values
    mask = df[color_col] == 'None'

    # Scatter plot for regular data
    scatter_fig = px.scatter(
        df[~mask],
        x=x_col,
        y=y_col,
        color=color_col,
        labels={"color": color_col},
        title='Completeness vs Contamination',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Add traces for placeholder values, if any exist
    if mask.any():
        scatter_fig.add_trace(go.Scatter(
            x=df[mask][x_col],
            y=df[mask][y_col],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Missing'
        ))

    scatter_fig.update_layout(plot_bgcolor='white')
    return scatter_fig


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

# check whether the column is numeric
def is_numeric(series):
    """ Check if a pandas series is numeric. """
    return pd.api.types.is_numeric_dtype(series)

# Define taxonomic ranks
tax_ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

# Split the classification column into separate columns for each level
df[tax_ranks] = df['classification'].str.split(';', expand=True)

metadata_columns = [col for col in df.columns if col not in tax_ranks + ['classification', 'Genome']]
dropdown_options = [{'label': col, 'value': col} for col in metadata_columns if col != 'formatted_classification']

# create sunburst plot 
def create_sunburst(df, selected_metadata):
    path = tax_ranks + [selected_metadata]
    sunburst_fig = px.sunburst(
        df, 
        path=path,
        color=selected_metadata,
        
        color_continuous_scale=px.colors.sequential.RdBu,
        maxdepth=8 # This will show up to species level (6) plus one more for metadata
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
### app.layout region 
### app.layout region 
app.layout = html.Div([
    html.H1('metaFun : genome selector for COMPARATIVE_ANNOTATION', style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H2('Quality and number of all genomes', style={'textAlign': 'left', 'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.Label('Select your metadata', style={'fontSize': 15, 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=[{'label': col, 'value': col} for col in metadata_columns if col != 'formatted_classification'],
                        value=metadata_columns[0] if metadata_columns else None,
                        placeholder="Select metadata",
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px', 'verticalAlign': 'top'}),
                html.Div([
                    html.Label('Select taxonomy rank', style={'fontSize': 15, 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='tax-rank-dropdown',
                        options=[{'label': rank, 'value': rank} for rank in tax_ranks],
                        value=tax_ranks[0],
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px', 'verticalAlign': 'top'}),
                html.Div([
                    html.Label('Select taxonomy values', style={'fontSize': 15, 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='tax-value-dropdown',
                        multi=True,
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'marginBottom': '20px', 'textAlign': 'left'}),
        ], style={'backgroundColor': '#f0f0f0', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'}),


        html.Div([
            dcc.Graph(id='overall-scatter-plot', selectedData={'points': []},    config={
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                'scrollZoom': True
            },
            style={'height': '600px'}
        )
            #dcc.Graph(id='overall-scatter-plot', style={'height': '600px'})
        ], style={'width': '65%', 'display': 'inline-block', 'overflow': 'visible'}),
        html.Div([
            dcc.Graph(id='overall-distribution-plot', style={'height':'600px'})
        ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top', 'overflow': 'visible'}),
    ], style={**common_style, 'overflow': 'visible'}),    

    html.Div([
        html.Div([
            html.H2('Select genomes that you are interested in', style={'textAlign': 'left'}),

            html.Label('Select your metadata for subsetting genomes', style={'fontSize': 15, 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='metadata-dropdown',
                    options=dropdown_options,
                    value='pass.GUNC',  # Default value
                    style={'width': '300px'}
                ),
            html.Div([
                # Boolean filter
                dcc.Checklist(
                    id='bool-checklist',
                    options=[
                        {'label': 'True', 'value': True},
                        {'label': 'False', 'value': False},
                        {'label': 'None', 'value': 'None'}
                    ],
                    value=[True, False, 'None'],
                    style={'display': 'none'}
                ),
                # Numeric filter
                html.Div([
                    html.Div([
                        dcc.RangeSlider(
                            id='numeric-slider',
                            min=0,
                            max=100,
                            step=1,
                            marks={0: '0', 100: '100'},
                            value=[0, 100]
                        ),
                    ], id='numeric-slider-container', style={'display': 'none'}),
                    dcc.Checklist(
                        id='numeric-none-checklist',
                        options=[{'label': 'Include None', 'value': 'None'}],
                        value=['None']
                    )
                ], id='numeric-filter',style={'display': 'none'}),
                # Categorical filter
                dcc.Dropdown(
                    id='categorical-dropdown',
                    multi=True,
                    style={'display': 'none'}
                )
            ], id='dynamic-filtering')
                
            ],style={ 'width':'100%','margin': '10px auto','textAlign': 'left'}),
        ], style=section_style),#style={'padding': '20px'}),

        html.Div([
            html.H3('Sunburst Plot of Genome Distribution', style={'textAlign': 'center', 'marginBottom': '10px'}),            
            html.Div([
                html.Button('Reset to Top Level', id='reset-sunburst-button', n_clicks=0),
                html.Div(id='sunburst-level-display', children='You are seeing now: Entire Dataset')
            ], style={'margin-bottom': '10px'}),
            html.Div([
                dcc.Graph(id='sunburst-plot')
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '100%'}),
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
            html.Button('Save to Server', id='save-server-button', n_clicks=0),
            html.Div(id='save-status', style={
                'display': 'inline-block',
                'marginLeft': '10px',
                'color':'red',
                'fontWeight': 'bold',
                })

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
            data=df.head(20).to_dict('records'),
            page_size=20,
            style_table={'overflowX': 'scroll'},
            css=[{ 
                'selector': '.dash-spreadsheet-page-selector',
                'rule': 'display: flex; justify-content: flex-start; margin-left: 10px;'

                }],  # Enable horizontal scroll
            filter_action='custom',
            filter_query='',
            sort_action='custom',
            sort_mode='multi',
            sort_by=[]
        ),
        dcc.Download(id="download-dataframe-csv"),
    ], style=common_style)
])
### app.layout region 
### app.layout region 
###################


def is_bool(series):
    return set(series.dropna().unique()).issubset({True, False})


@app.callback(
    [Output('bool-checklist', 'style'),
     Output('bool-checklist', 'options'),
     Output('bool-checklist', 'value'),
     Output('numeric-slider-container', 'style'),
     Output('numeric-slider', 'min'),
     Output('numeric-slider', 'max'),
     Output('numeric-slider', 'value'),
     Output('numeric-slider', 'marks'),
     Output('numeric-filter', 'style'),
     Output('numeric-none-checklist', 'value'),
     Output('categorical-dropdown', 'style'),
     Output('categorical-dropdown', 'options'),
     Output('categorical-dropdown', 'value')],
    [Input('metadata-dropdown', 'value')]
)
def update_dynamic_filter(selected_metadata):
    if selected_metadata:
        column = df[selected_metadata]
        column = column.replace('None', np.nan)  # Replace 'None' with np.nan temporarily for checks
                
        if is_bool(column):
            options = [{'label': str(i), 'value': i} for i in [True, False]]
            if column.isnull().any():
                options.append({'label': 'None', 'value': 'None'})
            values = [True, False]
            if column.isnull().any():
                values.append('None')
            return ({'display': 'block'}, options, values,
                    {'display': 'none'}, 0, 1, [0, 1], {},
                    {'display': 'none'}, [],
                    {'display': 'none'}, [], [])
        elif is_numeric(column):
            min_value = column.min()
            max_value = column.max()
            marks = {i: str(i) for i in range(int(min_value), int(max_value) + 1, (int(max_value) - int(min_value)) // 5)}
            return ({'display': 'none'}, [], [],
                    {'display': 'block'}, min_value, max_value, [min_value, max_value], marks,
                    {'display': 'block'}, ['None'] if column.isnull().any() else [],
                    {'display': 'none'}, [], [])
        else:
            options = [{'label': str(i), 'value': i} for i in column.fillna('None').unique()]
            return ({'display': 'none'}, [], [],
                    {'display': 'none'}, 0, 1, [0, 1], {},
                    {'display': 'none'}, [],
                    {'display': 'block'}, options, [o['value'] for o in options])
    return ({'display': 'none'}, [], [],
            {'display': 'none'}, 0, 1, [0, 1], {},
            {'display': 'none'}, [],
            {'display': 'none'}, [], [])


@app.callback(
    Output('tax-value-dropdown', 'options'),
    [Input('tax-rank-dropdown', 'value')]
)
def update_tax_value_options(selected_tax_rank):
    if not selected_tax_rank:
        return []
    unique_values = df[selected_tax_rank].unique()
    return [{'label': val, 'value': val} for val in unique_values if val != 'None']


@app.callback(
    [Output('overall-scatter-plot', 'figure'),
     Output('overall-distribution-plot', 'figure')],
    [Input('color-dropdown', 'value'),
     Input('tax-rank-dropdown', 'value'),
     Input('tax-value-dropdown', 'value'),
     Input('overall-scatter-plot', 'selectedData')]
)

def update_overall_plots(color_var, selected_tax_rank, selected_tax_values, selected_data):

#def update_overall_plots(color_var, selected_tax_rank, selected_tax_values):
    # Create a copy of the dataframe for plot manipulation
    plot_df = df.copy()
    # Convert 'None' placeholders to -999 only for the plotting data
    plot_df[color_var] = plot_df[color_var].replace('None', -999)

    if selected_tax_rank and selected_tax_values:
        plot_df = plot_df[plot_df[selected_tax_rank].isin(selected_tax_values)]
        
 
    # Mask for identifying placeholders in the dataset
    placeholder_mask = plot_df[color_var] == -999



    if selected_data and selected_data['points']:
        selected_points = selected_data['points']
        selected_indices = [point['pointIndex'] for point in selected_points]
        filtered_data = plot_df.iloc[selected_indices]
    else:
        filtered_data = plot_df


    # Create the scatter plot for actual data
    scatter_fig = px.scatter(
        filtered_data[~placeholder_mask],  # Use only valid data points
        x='Completeness',
        y='Contamination',
        color=color_var,
        hover_data=['formatted_classification'],
        title='Overall: Completeness vs Contamination',
        labels={color_var: color_var},  # Ensure the color variable is labelled correctly in the legend
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Add traces for placeholder values, if any
    if placeholder_mask.any():
        scatter_fig.add_trace(go.Scatter(
            x=plot_df[placeholder_mask]['Completeness'],
            y=plot_df[placeholder_mask]['Contamination'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x', line=dict(color='Black', width=1)),
            name='Not available'
        ))

    scatter_fig.update_layout(
        plot_bgcolor='white', font={'family': 'Arial', 'size': 14},
        dragmode='select',
        clickmode='event+select',
        selectdirection='any',
         legend=dict(
            title='',
            orientation='h',
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1)
    )

    # Distribution plot
    if selected_data and selected_data['points']:
        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        df_for_hist = plot_df.iloc[selected_indices]
    else:
        df_for_hist = plot_df

    df_for_hist[color_var] = df_for_hist[color_var].replace(-999, 'Not available')
    
    if is_numeric(df_for_hist[color_var]):
        dist_fig = px.histogram(
            df_for_hist[df_for_hist[color_var] != 'Not available'],
            x=color_var,
            title=f'Distribution of {color_var} for Selected Points',
            nbins=30
        )
    else:
        value_counts = df_for_hist[color_var].value_counts()
        dist_fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution of {color_var} for Selected Points'
        )
        dist_fig.update_xaxes(title_text=color_var)
        dist_fig.update_yaxes(title_text='Number of genomes')

    not_available_count = (df_for_hist[color_var] == 'Not available').sum()
    if not_available_count > 0:
        dist_fig.add_annotation(
            x=0.5, y=1.05,
            xref='paper', yref='paper',
            text=f'Number of genomes without metadata: {not_available_count}',
            showarrow=False,
            yshift=10,
            font=dict(color='red', size=12)
        )

    dist_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    return scatter_fig, dist_fig    


@app.callback(
    Output('sunburst-level-display', 'children'),
    [Input('reset-sunburst-button', 'n_clicks'),
     Input('sunburst-plot', 'clickData')]
)
def update_sunburst_level(reset_clicks, clickData):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'You are seeing now: Entire Dataset'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-sunburst-button' or not clickData:
        return 'You are seeing now: Entire Dataset'
    else:
        clicked_level = clickData['points'][0]['label']
        return f'You are seeing now: {clicked_level}'



@app.callback(
    Output('sunburst-plot', 'figure'),
    [Input('metadata-dropdown', 'value'),
     Input('bool-checklist', 'value'),
     Input('numeric-slider', 'value'),
     Input('numeric-none-checklist', 'value'),
     Input('categorical-dropdown', 'value'),
     Input('reset-sunburst-button', 'n_clicks')],
    [State('sunburst-plot', 'clickData')]
)
def update_sunburst(selected_metadata, bool_values, numeric_range, include_none, categorical_values, reset_clicks, click_data):
#def update_sunburst(selected_metadata, bool_values, numeric_range, include_none, categorical_values):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-sunburst-button' or not click_data:
        filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        if click_data:
            current_path = click_data['points'][0]['id'].split('/')
            for i, level in enumerate(current_path):
                if level and i < len(tax_ranks):
                    filtered_df = filtered_df[filtered_df[tax_ranks[i]] == level]




    if not selected_metadata:
        return go.Figure()  # Return an empty figure if no metadata is selected

    filtered_df = df.copy()
    column = filtered_df[selected_metadata]

    if is_bool(column):
        if 'None' in bool_values:
            filtered_df = filtered_df[(column.isin(bool_values)) | (column.isnull())]
        else:
            filtered_df = filtered_df[column.isin(bool_values)]
    elif is_numeric(column):
        if include_none and 'None' in include_none:
            filtered_df = filtered_df[
                ((column >= numeric_range[0]) & (column <= numeric_range[1])) | (column.isnull())
            ]
        else:
            filtered_df = filtered_df[
                (column >= numeric_range[0]) & (column <= numeric_range[1])
            ]
    else:  # categorical
        if 'None' in categorical_values:
            filtered_df = filtered_df[(column.isin(categorical_values)) | (column.isnull())]
        else:
            filtered_df = filtered_df[column.isin(categorical_values)]

    return create_sunburst(filtered_df, selected_metadata)






@app.callback(
    [Output('filtered-scatter-plot', 'figure'),
     Output('filtered-distribution-plot', 'figure')],
    [Input('sunburst-plot', 'clickData'),
     Input('metadata-dropdown', 'value'),
     Input('bool-checklist', 'value'),
     Input('numeric-slider', 'value'),
     Input('numeric-none-checklist', 'value'),
     Input('categorical-dropdown', 'value')],
    [State('sunburst-plot', 'figure')]
)

def update_filtered_plots(clickData, selected_metadata, bool_values, numeric_range, include_none, categorical_values, current_figure):
#def update_filtered_plots(clickData, selected_metadata, current_figure):
    filtered_df = df.copy()
    
    if clickData:
        current_path = clickData['points'][0]['id'].split('/')
        for i, level in enumerate(current_path):
            if level and i < len(tax_ranks):
                filtered_df = filtered_df[filtered_df[tax_ranks[i]] == level]
    column = filtered_df[selected_metadata]
    if is_bool(column):
        if 'None' in bool_values:
            filtered_df = filtered_df[(column.isin(bool_values)) | (column.isnull())]
        else:
            filtered_df = filtered_df[column.isin(bool_values)]
    elif is_numeric(column):
        if include_none and 'None' in include_none:
            filtered_df = filtered_df[
                ((column >= numeric_range[0]) & (column <= numeric_range[1])) | (column.isnull())
            ]
        else:
            filtered_df = filtered_df[
                (column >= numeric_range[0]) & (column <= numeric_range[1])
            ]
    else:  # categorical
        if 'None' in categorical_values:
            filtered_df = filtered_df[(column.isin(categorical_values)) | (column.isnull())]
        else:
            filtered_df = filtered_df[column.isin(categorical_values)]

    
    if filtered_df.empty:
        return dash.no_update, dash.no_update
    
    hover_data = ['formatted_classification'] + metadata_columns
    filtered_df[selected_metadata] = filtered_df[selected_metadata].replace('None', np.nan)
    is_numeric_metadata = pd.api.types.is_numeric_dtype(filtered_df[selected_metadata])
    filtered_df[selected_metadata] = filtered_df[selected_metadata].fillna('None')

    # Replace 'None' with np.nan temporarily for numeric check
    filtered_df[selected_metadata] = filtered_df[selected_metadata].replace('None', np.nan)

    # Check if the selected metadata is numeric
    is_numeric_metadata = pd.api.types.is_numeric_dtype(filtered_df[selected_metadata])

    # Replace np.nan back to 'None' for consistency
    filtered_df[selected_metadata] = filtered_df[selected_metadata].fillna('None')

    if is_numeric_metadata:
        # If numeric, replace 'None' with -999 for plotting
        plot_df = filtered_df.copy()
        plot_df[selected_metadata] = plot_df[selected_metadata].replace('None', -999)
        
        # Create a mask for placeholder values
        placeholder_mask = plot_df[selected_metadata] == -999
        
        # Scatter plot
        scatter_fig = px.scatter(
            plot_df[~placeholder_mask],
            x='Completeness',
            y='Contamination',
            color=selected_metadata,
            hover_data=hover_data,
            title='Filtered: Completeness vs Contamination',
            labels={selected_metadata: selected_metadata},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        # Add placeholder trace if exists
        if placeholder_mask.any():
            scatter_fig.add_trace(go.Scatter(
                x=plot_df[placeholder_mask]['Completeness'],
                y=plot_df[placeholder_mask]['Contamination'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x', line=dict(color='Black', width=1)),
                name='Not available'
            ))
        scatter_fig.update_layout(
            plot_bgcolor='white', 
            font={'family': 'Arial', 'size': 14},
            legend=dict(
                title='',
                orientation='h',
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
            
        # Distribution plot
        dist_fig = px.histogram(
            plot_df[plot_df[selected_metadata] != -999],  # Exclude -999 from histogram
            x=selected_metadata,
            title=f'Filtered Distribution of {selected_metadata}',
            nbins=30
        )
        # Customize x-axis tick labels to show 'Not available' for -999
        not_available_count = placeholder_mask.sum()
        
        if not_available_count > 0:
            dist_fig.add_annotation(
                x=0.5, y=1.05,  # Position at the top center of the plot
                xref='paper', yref='paper', 
                text=f'Number of genomes without metadata: {not_available_count}',
                showarrow=False,
                yshift=10,  # Adjust yshift for better positioning
                font=dict(color='red', size=12)
            )
        
        #dist_fig.update_xaxes(ticktext=np.where(dist_fig.data[0].x == -999, 'Not available', dist_fig.data[0].x), 
        #                      tickvals=dist_fig.data[0].x)
        
    else:
        # If not numeric, plot as categorical data
        scatter_fig = px.scatter(
            filtered_df,
            x='Completeness',
            y='Contamination',
            color=selected_metadata,
            hover_data=hover_data,
            title='Filtered: Completeness vs Contamination'
        )
        dist_fig = px.histogram(
            filtered_df,
            x=selected_metadata,
            title=f'Filtered Distribution of {selected_metadata}'
        )
        
    scatter_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})
    dist_fig.update_layout(plot_bgcolor='white', font={'family': 'Arial', 'size': 14})

    return scatter_fig, dist_fig

@app.callback(
    Output('data-table', 'data'),
    [Input('search-input', 'value'),
     Input('sunburst-plot', 'clickData'),
     Input('metadata-dropdown', 'value'),
     Input('data-table', 'sort_by')],
    [State('data-table', 'data')]
)
def update_table(search_value, clickData, selected_metadata, sort_by, current_data):
    filtered_df = df.copy()
    
    # Sunburst 선택에 따른 필터링
    if clickData:
        current_path = clickData['points'][0]['id'].split('/')
        for i, level in enumerate(current_path):
            if level and i < len(tax_ranks):
                filtered_df = filtered_df[filtered_df[tax_ranks[i]] == level]
    
    # 검색어에 따른 필터링
    if search_value:
        filtered_df = filtered_df[filtered_df.apply(lambda row: any(str(search_value).lower() in str(cell).lower() for cell in row), axis=1)]
    
    # 정렬 적용
    if sort_by:
        filtered_df = filtered_df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[col['direction'] == 'asc' for col in sort_by],
            inplace=False
        )
    
    return filtered_df.to_dict('records')



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
#    matched_df = df_raw[df_raw.index.isin(filtered_df.index)]
    if 'Genome' in filtered_df.columns:
        genome_ids = filtered_df['Genome'].tolist()
        matched_df = df_raw[df_raw['Genome'].isin(genome_ids)]
    

    return dcc.send_data_frame(matched_df.to_csv, f"{filename}.csv", index=False)




@app.callback(
    Output("save-status", "children"),
    Input("save-server-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_to_server(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    
    return "You can save it into your server. (Actual saving is not implemented here. Full path information appears.when you save file.)"
# callback for save into server 
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
    
    filtered_df = pd.DataFrame(table_data)
    
    if 'Genome' in filtered_df.columns:
        genome_ids = filtered_df['Genome'].tolist()
        matched_df = df_raw[df_raw['Genome'].isin(genome_ids)]
        
    server_path = os.path.join(os.getcwd(), f"{filename}.csv")
    matched_df.to_csv(server_path, index=False)
    return f"File saved to server at {server_path}"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metaFun: genome selector for COMPARATIVE_ANNOTATION')
    parser = argparse.ArgumentParser(description='refer to the documentation at https://metafun-doc-v01.readthedocs.io/en/latest/')
    parser.add_argument('-i', '--input', help='Input CSV file', required=True)
    args = parser.parse_args()

    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
    #app.run_server(debug=True)