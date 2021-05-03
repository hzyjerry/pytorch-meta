import gym
import ray
from assistive_gym.utils import save_agent

ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)

# Save as rllib model
class RLLibSaver(object):
    def __init__(self, envname="BedBathingJacoHuman-v0217_0-v1"):
        from assistive_gym.utils import load_policy_pair
        from dotmap import DotMap

        self.algo = "ppo"
        self.env = gym.make("assistive_gym:" + envname)

        policy_pair = DotMap({
            "human": [{
                "policy_path": None,
                "policy_coop": True,
                "policy_prob": 1.
            }],
            "robot_policy_path": None,
            "robot_policy_coop": True
        })
        self.agent, self.checkpoint_agent, mapping_fn, _ = load_policy_pair(self.env, self.algo, envname, policy_pair, coop=True, seed=1)


    def save(self, model, save_path, iteration):
        rllib_weights = self.agent.get_weights()
        maml_weights = model.state_dict()

        rllib_keys = [k for k in rllib_weights['human_0'].keys() if "value" not in k]
        maml_keys = [k for k in maml_weights.keys()]

        # Set weights
        for rkey, mkey in zip(rllib_keys, maml_keys):
            # Torch tranposes weights
            mweight = maml_weights[mkey].transpose(0, -1)
            assert rllib_weights['human_0'][rkey].shape == mweight.shape
            rllib_weights['human_0'][rkey] = mweight

        import pdb; pdb.set_trace()

        # Sync weights
        self.agent.set_weights(rllib_weights)
        self.agent.workers.foreach_worker(lambda ev: ev.set_weights(rllib_weights))
        import pdb; pdb.set_trace()

        checkpoint_agent._iteration = iteration
        checkpoint_path = save_agent(agent, checkpoint_agent, save_path, train_robot_only=False)
        return checkpoint_path
