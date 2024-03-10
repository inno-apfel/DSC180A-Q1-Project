#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/main')

import etl
import features
import models

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'all'. 
    '''

    with open('config/config.json', 'r') as fh:
        params = json.load(fh)

    run_all = False

    print()

    if 'all' in targets: # targets: data, features, models, forcast, clean, all
        run_all = True

    if run_all or ('data' in targets):
        print('currently running: data')
        etl.run(params)
        print('DONE')

    if run_all or ('features' in targets):
        print('currently running: features')
        features.run()
        print('DONE')

    if run_all or ('models' in targets):
        print('currently running: model training and evaluation')
        models.run()
        print('DONE')
        print()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)