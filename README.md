# Employment Growth Forecasting using U.S. Census Data

## Overview

This repository serves as a codebase for our machine learning employment forecasting models, developed in partnership with The San Diego Association of Government (SANDAG).


## Setting up the Enviroment

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`


## Retrieving the Data Locally:

Before running our forecasting models, you must set up a few required datasets. Our models are built on the [Census Bureau County Business Patterns](https://www.census.gov/programs-surveys/cbp/data/datasets.All.List_1222676053.html) zip-code industry details totals datasets, and compared against [SANDAG's Series 14 Forecasts on Jobs by ZIP Code](https://opendata.sandag.org/Forecast/Series-14-Forecasts-Jobs-by-ZIP-Code/gzcd-xn9p/about_data). For easy of use and reproducibility, a mirror of the datasets we used can be obtained [here](https://drive.google.com/file/d/1VYtJXJOHdor53l4ga_9xIvyeyWZfdBci/view?usp=sharing).

To access the included data, extract the compressed data into the directory `src/data`. The result should be the creation of the `raw` folder within the data directory.

Last Updated: 12/9/2023

  
## Running the Project

To use our forecasting models, run the `run.py` script from your terminal with the following targets:
- `data`: load and process the data according to `config.json`
  - by default, models are trained and evaluated on data for the San Diego region between 2012-2021
  - update `config.json` to include relavent zip-codes and years if you choose to explore different regions or year ranges.
- `features`: build the neccessary features for our models
- `forecast`:
  1. train our forecasting models
  2. evaluate them on a reserved subset of the data
  3. generate chloropleth maps of forecasted employement

  
## Contributors

- [Maximilian Wei](https://www.linkedin.com/in/maxhtwei/)