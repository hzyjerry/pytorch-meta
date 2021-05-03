import os
import gym
import torch
import logging
import torch.nn.functional as F

from tqdm import tqdm
from torchmeta.algos.maml.save import RLLibSaver
from torchmeta.toy.helpers import sinusoid, behaviour
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.algos.maml.model import ConvolutionalNeuralNetwork, PolicyNetwork

logger = logging.getLogger(__name__)




def train(inputs=[],
          test_inputs=[],
          exp_name="maml",
          batch_size=16,
          num_workers=1,
          use_cuda=False,
          num_batches=100,
          step_size=0.4,
          save_per=-1,
          learning_rate=0.01,
          first_order=False,
          seed=None,
          logger_kwargs={},
    ):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def get_loss(output, targets):
        """Compute the accuracy (after adaptation) of MAML on the test/query points

        Parameters
        ----------
        logits : `torch.FloatTensor` instance
            Outputs/logits of the model on the query points. This tensor has shape
            `(num_examples, num_classes)`.

        targets : `torch.LongTensor` instance
            A tensor containing the targets of the query points. This tensor has
            shape `(num_examples,)`.

        Returns
        -------
        accuracy : `torch.FloatTensor` instance
            Mean accuracy on the query points
        """
        return -1 * output.log_prob(targets).sum(dim=1).mean()

    from torch.utils.tensorboard import SummaryWriter
    import datetime

    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

    test_inputs = []
    env_name = "BedBathingBaxterHuman-v0217_0-v1"
    env = gym.make('assistive_gym:'+env_name)

    dataset = behaviour(inputs, shots=1000, test_shots=200)
    # dataset = sinusoid(shots=1000, test_shots=100)
    # model = PolicyNetwork(1, 1).double
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)

    model = PolicyNetwork(env.observation_space_human.shape[0], env.action_space_human.shape[0])
    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    now = datetime.datetime.now()
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    second = '{:02d}'.format(now.second)
    timestamp = '{}-{}-{}'.format(hour, minute, second)
    output_dir = logger_kwargs['output_dir']
    exp_folder = os.path.join(output_dir, "runs", f"{exp_name}_{timestamp}")
    print(f"Saving to {exp_folder}")
    os.makedirs(exp_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_folder, comment=f"{exp_name}")

    rllib_saver = RLLibSaver()
    # Training loop
    with tqdm(dataloader, total=num_batches, disable=True) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device).float()
            train_targets = train_targets.to(device=device).float()

            test_inputs, test_targets = batch['test']
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

            # pbar.update(1)
            pbar.set_postfix(loss='{0:.4f}'.format(loss.item()))
            print(f"Iter {batch_idx} train loss: {loss.item():.3f}")
            writer.add_scalar(f"train/maml_loss", loss.item(), batch_idx)
            writer.flush()

            # Save model
            do_save = (batch_idx % save_per == 0) or (batch_idx == num_batches)
            if exp_folder is not None and save_per > 0 and do_save:
                rllib_saver.save(model, exp_folder, batch_idx)
                # filename = os.path.join(exp_folder, 'maml_behaviour_{num_shots:02d}.th')
                # print(f"Saving to {filename}")
                # rllib_env, rllib_agent = save_rllib_policy(model, rllib_env, rllib_agent)
                # with open(filename, 'wb') as f:
                #     state_dict = model.state_dict()
                #     torch.save(state_dict, f)

            if batch_idx >= num_batches:
                break
