<h1 align='left'>
  Forecasting Regional Aggregate Establishment Birth-Death Values: Using Algorithmic Modeling
</h1>

<h3 align="center">
  <br>
  <img src="https://datascience.ucsd.edu/wp-content/uploads/2022/09/UCSanDiego_Halicioglu_BlueGold.png", width=450>
  <br>
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/SANDAG_logo.png", width=450>
  <br>
</h3>



## Overview

This repository serves as a working codebase for our machine learning employment forecasting models, developed in partnership with The San Diego Association of Government (SANDAG). Using various python machine learning packages, we trained and evaluated candidate modeling architectures to forecast establishment growth in the San Diego County.

There are currently two ways you can interact with our work:

1. Interacting with our executable script `run.py` (recommended)
2. Exploring our work through our development interactive python notebooks
   - note disclosures in `notebooks/README.md`


## Setting up the Enviroment

To begin running our models, you must first replicate our virtual enviroment. To do so, follow the below steps:

1. Clone the repository and navigate to the project directory:

   ```
   git clone https://github.com/inno-apfel/regional-business-growth-forecasting.git
   cd regional-business-growth-forecasting
   ```

2. Create and activate a conda enviroment from the provided `enviroment.yml` file:

   ```
   conda env create -f enviroment.yml
   conda activate regional-business-growth-forecasting
   ```


## Retrieving the Data Locally:

Before running our forecasting models, you must set up a few required datasets. Our models are built on the [Census Bureau County Business Patterns](https://www.census.gov/programs-surveys/cbp/data/datasets.All.List_1222676053.html) zip-code industry details totals datasets, along with various [American Community Survey](https://www.census.gov/programs-surveys/acs/data.html) socio-demographic and economics datasets. For ease of use and reproducibility, a mirror of the datasets we used can be obtained [(here)](https://drive.google.com/file/d/16WWbarGoK95ily9YJRB3-RFuoBaZhvj7/view?usp=sharing).

To access the included data, extract the compressed data into the directory `src/data`. The result should be the creation of the `raw` folder within the data directory.

Last Updated: 03/10/2024

  
## Running the Project

To use our forecasting models, run the `run.py` script from your terminal with the following targets:
- `data`: load and process the data according to `config.json`
  - by default, models are trained and evaluated on data for the San Diego region between 2012-2021, recreating our results in `report.pdf`
  - update `config.json` to include relavent zip-codes and years if you choose to explore different regions or year ranges.
- `features`: build the neccessary features for our models
- `models`:
  1. train and evaluate our forecasting models for immediate-next-year and long-term forecasting
  2. generate comparison visualization of model forecasts on test data

- `forecast`:
  - generate chloropleth maps for model forecasts on last year in test data
  - generate region level forecasts from end of training data up to a user input year
    - uses our autoregressive feedback LSTM models
- `clean`:
  - removes all temporary files in the following directories:
     - `src/data/temp`
     - `out/forecast_tables`
     - `out/models`
     - `out/plots`
- `all`: run the above targets, except `clean`, in sequential order

notes: stable versions of saved models are included by default, run `clean` to delete them before re-training

  
## Contributors

- [Maximilian Wei](https://www.linkedin.com/in/maxhtwei/)
- [Mariana Montoya](https://www.linkedin.com/in/mariana-montoya11/)
- [Michael Lue](https://www.linkedin.com/in/michael-lue-6ba799201/)
