#!/usr/bin/env python
"""
This runs a DE configuration on all the 2-, 3-, 5-, 10-, 20-, 40-dimensional 24 BBOB test functions. This is based on "example_experiment_for_beginners.py" originally provided by the COCO framework (https://github.com/numbbo/coco).

The following is an example to run a DE with the synchronous model and the hand-tuned parameters:

python de_bbob.py -de_alg 'syn_de' -out_folder 'Syn' -archive_rate '1.0' -de_cr '0.9' -de_sf '0.5' -de_strategy 'rand_to_pbest_1' -p_best_rate '0.05' -pop_size_rate '13.0' -subset_size_rate '0.0' -children_size_rate '0.0'
"""

#import cocoex, cocopp  # experimentation and post-processing modules
import cocoex  # only experimentation module
import numpy as np
import sys
import random
import click
from de import SynchronousDE
from de import jDE
from de import JADE
from de import SHADE
from de import CoBiDE
from de import EPSDE
from de import DERSF
from de import SinDE
from de import CoDE
from de import cDE
from de import DECaRS

@click.command()
@click.option('--de_alg', required=True, default='syn_de', type=str, help='DE algorithm')
@click.option('--jade_learning_rate', required=False, default=0.5, type=float, help='Learning rate (c) in JADE')
@click.option('--shade_memory_size', required=False, default=10, type=int, help='Memory size (H) in SHADE')
@click.option('--pop_size', required=False, default=100, type=int, help='Population size')
@click.option('--de_sf', required=False, default=0.5, type=float, help='Scale factor in DE')
@click.option('--de_cr', required=False, default=0.9, type=float, help='Crossover rate in DE')
@click.option('--de_strategy', required=True, default='rand_1', type=str, help='Differential mutation strategy')
@click.option('--p_best_rate', required=False, default=0.05, type=float, help='P-best rate in p-best-based mutation strategies')
@click.option('--archive_rate', required=False, default=1.0, type=float, help='Archive size rate in p-best-based mutation strategies')
@click.option('--success_criterion', required=False, default='conventional', type=str, help='Type of the success criterion for parameter control')
@click.option('--int_handling', required=False, default='Baldwin', type=str, help='Type of method for handling integer variables')
@click.option('--budget_multiplier', required=False, default=10000, type=int, help='A budget multiplier that determins the maximum number of function evaluations.')
@click.option('--output_folder', required=True, default='tmp', type=str, help='Output folder')
def run(de_alg, jade_learning_rate, shade_memory_size, pop_size, de_sf, de_cr, de_strategy, p_best_rate, archive_rate, success_criterion, int_handling, budget_multiplier, output_folder):
    np.random.seed(seed=1)
    random.seed(1)
    
    suite_name = "bbob-mixint"
    # the maximum number of function evaluations is "budget_multiplier * dimensionality"
    use_logger = False

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        remaining_evals = problem.dimension * budget_multiplier
    
        de = None
        if de_alg == 'syn_de':
            de = SynchronousDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)          
        elif de_alg == 'jde':        
            de = jDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'jade':        
            de = JADE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'shade':        
            de = SHADE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'cobide': 
            de = CoBiDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'epsde':        
            de = EPSDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'dersf':        
            de = DERSF(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'sinde':        
            de = SinDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'code': 
            de = CoDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'cde':        
            de = cDE(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        elif de_alg == 'decars':        
            de = DECaRS(problem, problem.dimension, problem.lower_bounds, problem.upper_bounds, remaining_evals, pop_size, de_strategy, de_sf, de_cr, p_best_rate, archive_rate, success_criterion, int_handling, use_logger)
        else:
            raise Exception('Error: %s is not defined' % de_alg)        
        de.run()       
        minimal_print(problem, final=problem.index == len(suite) - 1)

if __name__ == '__main__':
    run()        
