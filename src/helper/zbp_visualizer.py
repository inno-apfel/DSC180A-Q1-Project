import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px
from IPython.display import IFrame

def generate_zbp_chloropleth(data, group, value, outpath):
    """
    Generates a chloropleth map of the USA by ZIP Code

    Parameters
    ----------
    data: pandas.DataFrame
        The data used to generate a chloropleth.
    group: str
        The name of the column in data denoting ZIP Code.
    value: str
        The name of the column in data denoting the value you want to display.
    outpath: str
        The filepath to save the generated plot at.
        
    Out
    ---
    zbp_plot_{tag}: HTML file
        The generated chloropleth map
    """
    df = data.groupby(group)[value].sum().reset_index()[[group, value]]
    
    with urlopen('https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ca_california_zip_codes_geo.min.json') as response:
        zipcodes = json.load(response)
    
    fig = px.choropleth(df, 
                        geojson=zipcodes, 
                        locations=group, 
                        color=value,
                        color_continuous_scale='blues',
                        range_color=(df[value].min(),df[value].max()),
                        featureidkey="properties.ZCTA5CE10",
                        scope="usa",
                        labels={'Final_Labels':'Cluster_Category'})
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.write_html(outpath)