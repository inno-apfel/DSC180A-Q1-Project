#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/main')

import etl
import features
import forecast

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'all'. 
    '''

    with open('config/config.json', 'r') as fh:
        params = json.load(fh)

    run_all = False

    print()

    if 'all' in targets: # targets: data, features, forecast, test, all, clean
        run_all = True

    if run_all or ('data' in targets):
        print('current running: data')
        etl.run(params)
        print('DONE')

    if run_all or ('features' in targets):
        print('current running: features')
        features.run()
        print('DONE')

    if run_all or ('forecast' in targets):
        print('current running: forecast')
        print()
        print('-----------------------------------------------------------')
        print(f"Models will be trained on data from {min(params['years'])} to <input-year>")
        print(f"Models will be evaluated on data from <input-year> to {max(params['years'])}")
        print(f"Prediction visualizations will be generated for {max(params['years'])}")
        print(f"Current year range for available data: {min(params['years'])} - {max(params['years'])}")
        print('-----------------------------------------------------------')
        end_year = int(input('(input-year): '))
        forecast.run(end_year)
        print()
        print('View model prediction visualizations in out/plots')
        print()
        print('DONE')
        print()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)