#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/code')

import code

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'all'. 
    '''
    if 'all' in targets: # potential targets: data, analysis, model, test, all, clean
        with open('config/config.json', 'r') as fh:
            params = json.load(fh)
        code.run_process(**params)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)