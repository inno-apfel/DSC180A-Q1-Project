# DSC180A Q1 Project: Business Growth Forecasting Research

__This repository displays__: the intended structure of our codebase


## Retrieving the data locally:

(1) Download the data file **zbp_data.rar** from DSMLP team b10 group directory

(2) Extract all into src/data/raw


## Running the project

* To install the dependencies, run the following command from the root directory of the project: `pip install -r requirements.txt`

  
### Building the project stages using `run.py`

* To get the data, from the project root dir, run `python run.py data features`
  - This fetches the data, creates features, cleans data and saves the data in data/temp directory.
  - NOT YET FUNCTIONAL
* To get the results of statistical test, from the project root dir, run `python run.py all`
  - This fetches the data, creates the features, and outputs a temporary 'success' message.
  
## Reference
N/A