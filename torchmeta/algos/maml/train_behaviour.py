import os
import gym
import yaml
import torch
import logging
import torch.nn.functional as F
from dotmap import DotMap
from tqdm import tqdm
from torchmeta.algos.maml.save import RLLibSaver
from torchmeta.algos.maml.eval import *
from torchmeta.toy.helpers import sinusoid, behaviour
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.algos.maml.model import ConvolutionalNeuralNetwork, PolicyNetwork
from collections import OrderedDict

logger = logging.getLogger(__name__)




def train(inputs=[],
          adapt_inputs=[],
          exp_name="maml",
          batch_size=16,
          num_workers=1,
          use_cuda=False,
          num_batches=100,
          step_size=0.4,
          shots=1000,
          test_shots=200,
          save_per=-1,
          eval_per=1,
          learning_rate=0.01,
          first_order=False,
          save_dir=".",
          date="210101",
          seed=None,
          logger_kwargs={},
          exp_params=DotMap()
    ):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def get_loss(output, targets):
        return -1 * output.log_prob(targets).sum(dim=1).mean()

    def get_adapted_params(model, test_batch):
        test_inputs, test_targets = test_batch['train']
        test_in, test_target = test_inputs[0], test_targets[0]
        test_out = model(test_in)
        inner_loss = get_loss(test_out, test_target)
        model.zero_grad()
        with torch.no_grad():
            params = gradient_update_parameters(model, inner_loss, step_size=step_size, first_order=first_order)
            test_out = model(test_in, params=params)
            outer_loss = get_loss(test_out, test_target)
        return params, outer_loss.item()

    from torch.utils.tensorboard import SummaryWriter
    import datetime


    env_name = "BedBathingBaxterHuman-v0217_0-v1"
    env = gym.make('assistive_gym:'+env_name)

    dataset = behaviour(inputs, shots=shots, test_shots=test_shots)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)

    adapt_datasets, adapt_loaders = {}, {}
    for key, adapt_dir in adapt_inputs.items():
        adapt_datasets[key] = behaviour([adapt_dir], shots=shots, test_shots=test_shots)
        adapt_loaders[key] = BatchMetaDataLoader(adapt_datasets[key], batch_size=1, shuffle=True, num_workers=num_workers)

    model = PolicyNetwork(env.observation_space_human.shape[0], env.action_space_human.shape[0])
    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    now = datetime.datetime.now()
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    second = '{:02d}'.format(now.second)
    timestamp = '{}-{}-{}'.format(hour, minute, second)

    # new_data/date/MAML_assistive_gym_sz0-1_lr0-01_s50_ts200/MAML_assistive_gym_sz0-1_lr0-01_s50_ts200_s0
    output_dir = logger_kwargs['output_dir']
    log_folder = os.path.join(save_dir, date, "runs", f"{exp_name}_{timestamp}")
    print(f"Saving logs to {log_folder}")
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder, comment=f"{exp_name}")

    rllib_saver = RLLibSaver()
    adapt_losses = {key: [] for key in adapt_loaders.keys()}
    # Training loop
    with tqdm(dataloader, total=num_batches, disable=True) as pbar:
        for batch_idx, batches in enumerate(zip(pbar, *list(adapt_loaders.values()))):
            model.zero_grad()

            main_batch = batches[0]
            train_inputs, train_targets = main_batch['train']
            train_inputs = train_inputs.to(device=device).float()
            train_targets = train_targets.to(device=device).float()

            test_inputs, test_targets = main_batch['test']
            test_inputs = test_inputs.to(device=device).float()
            test_targets = test_targets.to(device=device).float()

            outer_loss = torch.tensor(0., device=device)
            loss = torch.tensor(0., device=device)
            for task_idx, (train_input, train_target, test_input, test_target) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
                train_output = model(train_input)
                inner_loss = get_loss(train_output, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=step_size,
                                                    first_order=first_order)

                test_output = model(test_input, params=params)
                outer_loss += get_loss(test_output, test_target)
                with torch.no_grad():
                    loss += get_loss(test_output, test_target)

            outer_loss.div_(batch_size)
            loss.div_(batch_size)

            outer_loss.backward()
            meta_optimizer.step()

            # Report progress
            pbar.set_postfix(loss='{0:.4f}'.format(loss.item()))
            print(f"Iter {batch_idx} train loss: {loss.item():.3f}")
            writer.add_scalar(f"train/maml_loss", loss.item(), batch_idx)
            writer.flush()

            # Eval & Save model
            do_eval = batch_idx % eval_per == 0
            do_save = output_dir is not None and save_per > 0 and ((batch_idx % save_per == 0) or (batch_idx == num_batches))
            if do_eval:
                all_pre_params = []
                all_post_params = []
                all_inputs = []

                for adapt_key, adapt_batch in zip(list(adapt_loaders.keys()), batches[1:]):

                    pre_params = OrderedDict(model.meta_named_parameters())
                    post_params, outer_loss = get_adapted_params(model, adapt_batch)

                    all_pre_params.append(pre_params)
                    all_post_params.append(post_params)
                    all_inputs.append(adapt_batch['train'][0][0]) # inputs, idx=1
                    print(f"Save inner loss {adapt_key}: {outer_loss:.04f}")
                    if do_save:
                        rllib_saver.save(params=post_params, save_path=output_dir, key=adapt_key, iteration=batch_idx, exp_params=exp_params)
                    adapt_losses[adapt_key].append(outer_loss)
                adapt_keys = list(adapt_loaders.keys())
                if len(adapt_keys) > 1:
                    _, _, fig = cluster_activation(model, all_inputs, all_pre_params, adapt_keys, "fc3")
                    writer.add_figure(f"train/fc3_before", fig, batch_idx)
                    writer.flush()
                    _, _, fig = cluster_activation(model, all_inputs, all_post_params, adapt_keys, "fc3")
                    writer.add_figure(f"train/fc3_after", fig, batch_idx)
                    writer.flush()
                with open(os.path.join(output_dir, "adapt_losses.txt"), "w+") as f:
                    yaml.dump(adapt_losses, f)

            if batch_idx >= num_batches:
                break
