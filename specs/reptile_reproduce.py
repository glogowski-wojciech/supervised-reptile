from mrunner.experiment import Experiment
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'specs'))
from spec_utils import get_git_head_info, get_combinations
# It might be a good practice to not change specification files if run
# successfully, to keep convenient history of experiments. When you want to run
# the same experiment with different hyper-parameters, just copy it.
# Starting name with (approximate) date of run is also helpful.

def create_experiment_for_spec(parameters):
    script = 'run_omniglot.py'
    # this will be also displayed in jobs on prometheus
    name = 'wg_test'
    project_name = "deepsense-ai-research/meta-learning-reptile-dirty"
    python_path = '.:specs'
    paths_to_dump = ''  # e.g. 'plgrid tensor2tensor', do we need it?
    tags = 'mrunner debug'.split(' ')
    parameters['git_head'] = get_git_head_info()
    return Experiment(project=project_name, name=name, script=script,
                      parameters=parameters, python_path=python_path,
                      paths_to_dump=paths_to_dump, tags=tags,
                      time='0-0:5'  # days-hours:minutes # TODO
                      )

# Set params_configurations, eg. as combinations of grid.
# params are also good place for e.g. output path, or git hash
params_grid = dict(
    dataset=['/net/archive/groups/plggluna/wglogowski/tensorflow/omniglot'],
    mode=['o15'],
    eval_interval=[100],
    debug=[True],
)
params_configurations = get_combinations(params_grid)


def spec():
    experiments = [create_experiment_for_spec(params)
                   for params in params_configurations]
    return experiments
