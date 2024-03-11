#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/main')
sys.path.insert(0, 'src/helper')

import pandas as pd

from animation_loader import AnimationLoader

import etl
import features
import models
import forecast
import clean

import time

def print_df_indented(df_string, indent):
    end_of_col_names = df_string.index('\n')
    col_names = df_string[:end_of_col_names]
    remaining_rows = df_string[end_of_col_names + 1:]
    print(indent, col_names)
    print(indent, '-'*len(col_names))
    for row in remaining_rows.split('\n'):
        print(indent, row)


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'all'. 
    '''

    spinner_animation = AnimationLoader()
    indent = '    '

    with open('config/config.json', 'r') as fh:
        params = json.load(fh)

    run_all = False
    if 'all' in targets: # targets: data, features, models, forcast, clean, all
        run_all = True

    if run_all or ('data' in targets):
        curr_task = 'loading data:'
        print()
        spinner_animation.show(curr_task, finish_message=f'{curr_task} done', failed_message=f'{curr_task} failed')
        etl.run(params)
        spinner_animation.finished = True
        print()
        print(indent, "loaded data located at 'src/data/temp'")

    if run_all or ('features' in targets):
        curr_task = 'transforming features:'
        print()
        spinner_animation.show(curr_task, finish_message=f'{curr_task} done', failed_message=f'{curr_task} failed')
        features.run()
        spinner_animation.finished = True
        print()
        print(indent, "proccessed data located at 'src/data/temp'")

    if run_all or ('models' in targets):
        curr_task = 'training-evaluating models:'
        print()
        spinner_animation.show(curr_task, finish_message=f'{curr_task} done', failed_message=f'{curr_task} failed')
        models.run()
        spinner_animation.finished = True
        print()
        model_evaluations_df = pd.read_csv('src/data/temp/model_evaluations.csv')
        model_evals_df_str = model_evaluations_df.to_string(index=False)
        print_df_indented(model_evals_df_str, indent)
        print()
        print(indent, "3-year out forecast located at 'out/plots/final_forecasts.jpg'")

    if run_all or ('forecast' in targets):
        curr_task = 'generating forecasts:'
        print()
        spinner_animation.show(curr_task, finish_message=f'{curr_task} done', failed_message=f'{curr_task} failed')
        forecast.run()
        spinner_animation.finished = True
        print()
        print(indent, "forecasts out to 2050 located at 'out/plots/feedback_2050_forecats.jpg'")
    
    if run_all or ('clean' in targets):
        curr_task = 'removing temporary files:'
        print()
        spinner_animation.show(curr_task, finish_message=f'{curr_task} done', failed_message=f'{curr_task} failed')
        clean.run()
        spinner_animation.finished = True

    print()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)