import os
from spinup.utils.run_utils import ExperimentGrid
from torchmeta.algos.maml.train_behaviour import train
from functools import partial
import torch


inputs = [
    "new_models/210429/dataset/BedBathingBaxterHuman-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210429/dataset/BedBathingJacoHuman-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210429/dataset/BedBathingPR2Human-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210429/dataset/BedBathingPandaHuman-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210429/dataset/BedBathingSawyerHuman-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210429/dataset/BedBathingStretchHuman-v0217_0-v1-human-coop-robot-coop_10k"
]
root = "/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/"
inputs = [root + inp for inp in inputs]



if __name__ == '__main__':
    import argparse, yaml
    from dotmap import DotMap

    parser = argparse.ArgumentParser()
    # parser.add_argument('params_file', type=str, default=None)
    # parser.add_argument('--cloud', default=False, action="store_true")
    # args = parser.parse_args()
    # assert args.params_file is not None
    # params = None
    # with open(args.params_file) as f:
    #     params = DotMap(yaml.load(f))
    batch_size = 16
    use_cuda = False
    num_batches = 2000
    # save_per = 200
    save_per = 1
    step_size = [0.1]
    learning_rate = [1e-2]
    save_dir = "data"
    date = "210502"
    exp_name = "MAML_assistive_gym"

    eg = ExperimentGrid(name=exp_name)
    eg.add('batch_size', batch_size)
    eg.add('num_batches', num_batches)
    eg.add('step_size', step_size, 'sz', True)
    eg.add('learning_rate', learning_rate, 'lr', True)

    eg.run(partial(train, exp_name=exp_name, batch_size=batch_size, use_cuda=use_cuda, num_batches=num_batches, inputs=inputs, save_per=save_per), num_cpu=1, data_dir=f"{save_dir}/{date}", pickle=False)
