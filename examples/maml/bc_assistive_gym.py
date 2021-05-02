import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import gym

from torchmeta.toy.helpers import sinusoid, behaviour
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from model import ConvolutionalNeuralNetwork, PolicyNetwork

logger = logging.getLogger(__name__)


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


def train(args):
    logger.warning('This script is an example to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

    inputs = [
        "new_models/210429/dataset/BedBathingBaxterHuman-v0217_0-v1-human-coop-robot-coop_10k"
    ]
    env_name = "BedBathingBaxterHuman-v0217_0-v1"
    env = gym.make('assistive_gym:'+env_name)

    dataset = behaviour(inputs, shots=400, test_shots=1)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model = PolicyNetwork(env.observation_space_human.shape[0], env.action_space_human.shape[0])
    for key, v in model.features.named_parameters():
        v.data = torch.nn.init.zeros_(v)
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    pbar = tqdm(total=args.num_batches)
    batch_idx = 0
    while batch_idx < args.num_batches:
        for batch in dataloader:
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device).float()
            train_targets = train_targets.to(device=args.device).float()

            loss = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target) in enumerate(zip(train_inputs, train_targets)):
                train_output = model(train_input)
                loss += get_loss(train_output, train_target)

            model.zero_grad()
            loss.div_(len(dataloader))
            loss.backward()
            meta_optimizer.step()

            pbar.update(1)
            pbar.set_postfix(loss='{0:.4f}'.format(loss.item()))
            batch_idx += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    train(args)
