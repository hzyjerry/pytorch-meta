from torchmeta.datasets.helpers import omniglot
from torchmeta.toy.helpers import sinusoid, behaviour
from torchmeta.utils.data import BatchMetaDataLoader

# dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

# for batch in dataloader:
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)



# dataset = sinusoid(shots=1000, test_shots=100)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

# for batch in dataloader:
#     train_inputs, train_targets = batch['train']
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)

inputs = [
    "new_models/210430/dataset/BedBathingJacoHuman-v0217_0-v1-human-coop-robot-coop_10k",
    "new_models/210430/dataset/BedBathingJacoHuman-v0217_0-v1-human-coop-robot-coop_10k"
]

dataset = behaviour(inputs, shots=1000, test_shots=200)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch['train']
    print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
