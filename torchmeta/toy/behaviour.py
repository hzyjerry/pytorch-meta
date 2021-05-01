import numpy as np
import os
import os.path as osp
from ray.rllib.offline import JsonReader, ShuffledInput
from torchmeta.utils.data import Task, MetaDataset


class Behaviour(MetaDataset):
    """
    Behaviour cloning dataset.

    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.

    """
    def __init__(self, inputs, num_samples_per_task=0, policy_id="human_0", transform=None, target_transform=None,
                 dataset_transform=None):
        super(Behaviour, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)

        self.transform = transform
        self._datasets = []
        for input_path in inputs:
            # Cache all samples in rllib, to ensure stochastic loading
            data_paths = [osp.join(input_path, f) for f in os.listdir(input_path)]
            # dataset = ShuffledInput(JsonReader(data_paths), n=num_samples_per_task)
            dataset = ShuffledInput(JsonReader(data_paths))
            self._datasets.append(dataset)

        self.num_tasks = len(inputs)
        self.num_samples_per_task = num_samples_per_task
        self.policy_id = policy_id


    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        dataset = self._datasets[index]
        num_samples = 0
        data = {'obs': [], 'actions': []}
        while num_samples < self.num_samples_per_task:
            dnext = dataset.next()
            acs = dnext.policy_batches[self.policy_id]['actions'] # (nsamples, acs_dim)
            obs = dnext.policy_batches[self.policy_id]['obs']
            data['obs'].append(obs)
            data['actions'].append(acs)
            num_samples += acs.shape[0]

        data['obs'] = np.concatenate(data['obs'])
        data['actions'] = np.concatenate(data['actions'])
        if num_samples > self.num_samples_per_task:
            data['obs'] = data['obs'][:self.num_samples_per_task]
            data['actions'] = data['actions'][:self.num_samples_per_task]
        # import pdb; pdb.set_trace()
        task = BehaviourTask(index, data, self.num_samples_per_task, self.transform, self.target_transform)


        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class BehaviourTask(Task):
    def __init__(self, index, data, num_samples, transform=None,
                 target_transform=None):
        super(BehaviourTask, self).__init__(index, None) # Regression task
        self._data = data
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self._data['obs'][index], self._data['actions'][index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (input, target)
