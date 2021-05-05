import numpy as np
from torchmeta.algos.maml.tsne import tsne


def cluster_activation(model, all_inputs, all_params, all_keys, layer="fc3", make_figure=True):
    """Cluster layer outputs for t-sne visualization.

    Params:
        all_inputs: list of inp, each (1, ninputs, ndim)

    """
    all_activations = []
    num_samples = 100
    for inp, params in zip(all_inputs, all_params):
        model(inp, params=params)
        # Average over samples
        activations = model.activation[layer].numpy()
        idx = np.random.choice(np.arange(len(activations)), num_samples, replace=False)
        activations = activations[idx]
        all_activations.append(activations)
    all_activations = np.concatenate(all_activations)
    tsne_out = tsne(all_activations, initial_dims=all_activations.shape[1])
    # print(f"Tsne out", tsne_out)
    if make_figure:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, txt in enumerate(all_keys):
            tsne_i = tsne_out[i*num_samples : (i+1)*num_samples]
            ax.scatter(tsne_i[:, 0], tsne_i[:, 1], label=txt)
        # for i, txt in enumerate(all_keys):
        #     ax.annotate(txt, (tsne_out[i, 0], tsne_out[i, 1]))
        ax.legend()
        return tsne_out, all_activations, fig
    else:
        return tsne_out, all_activations, None


def compute_similarity():
    """Compute layer similarity."""
    pass