from . import RLAgent
from common.registry import Registry
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent import utils
import gym
import os 

@Registry.register_model('tinylight')
class TinyLightAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.num_inters = len(self.world.intersections)
        self.model_dict = Registry.mapping['model_mapping']['setting'].param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_phase = self.action_space.n
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.obs_shape = [torch.Size([ob_generator.ob_length]) for ob_generator in self.ob_generators]
        self.obs_shape.append(torch.Size([self.num_phase])) if self.one_hot else self.obs_shape.append(torch.Size([1])) 

        self.replay_buffer = ReplayBuffer(Registry.mapping['trainer_mapping']['setting'].param['buffer_size'],
                                        Registry.mapping['model_mapping']['setting'].param['batch_size'],
                                        self.obs_shape,
                                        self.device)
        
        self.current_phase = 0
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']

        self.n_input_feature_dim = [ob_generator.ob_length for ob_generator in self.ob_generators]
        self.n_input_feature_dim.append(self.num_phase) if self.one_hot else self.n_input_feature_dim.append(1)
        self.n_layer_1_dim = [16, 18, 20, 22, 24]
        self.n_layer_2_dim = [16, 18, 20, 22, 24]
        self.n_alpha = torch.nn.ModuleList([
            Alpha(elem_size=len(self.n_input_feature_dim), config=self.model_dict),
            Alpha(elem_size=len(self.n_layer_1_dim), config=self.model_dict),
            Alpha(elem_size=len(self.n_layer_2_dim), config=self.model_dict)
        ])

        self.network_local = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.n_alpha,
            self.num_phase,
        ).to(self.device)
        self.network_target = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.n_alpha,
            self.num_phase,
        ).to(self.device)

        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=Registry.mapping['model_mapping']['setting'].param['learning_rate']
        )
        self.alpha_optim = optim.RMSprop(
            self.n_alpha.parameters(),
            lr=Registry.mapping['model_mapping']['setting'].param['learning_rate']
        )
        self.network_lr_scheduler = optim.lr_scheduler.StepLR(
            self.network_optim,
            step_size=10,
            gamma=0.5
        )

        self.beta = 16.0  # weight of alpha regularizer
        copy_model_params(self.network_local, self.network_target)
    
    def _reset_generator(self):
        super()._reset_generator()
        self.ob_generators = [LaneVehicleGenerator(self.world,  self.inter_obj, [ob_list], in_only=True, average=None) for ob_list in self.ob_list]
    
    def reset(self):
        super().reset()
        self.current_phase = 0
        
    def __repr__(self):
        return self.network_local.__repr__()
    
    def get_ob(self):
        x_obs = [ob_generator.generate() for ob_generator in self.ob_generators]
        # x_obs = [np.clip(ob_generator.generate(), a_min=None, a_max=10) for ob_generator in self.ob_generators]
        phase = np.array(self.phase_generator.generate())
        if self.one_hot:
            phase = utils.idx2onehot(phase, self.action_space.n)
        x_obs.append(phase)
        return x_obs
    
    def get_reward(self):
        rewards = self.reward_generator.generate()
        rewards = np.sum(rewards)
        return rewards
    
    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase
    
    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        observation = [torch.tensor(o, dtype=torch.float32, device=self.device) for o in ob]
        self.network_local.eval()
        b_q_value = self.network_local(observation)
        # action = torch.argmax(b_q_value, dim=1).cpu().item()
        action = torch.argmax(b_q_value).cpu().item()
        self.network_local.train()
        return action
    
    def sample(self):
        return np.random.randint(0, self.action_space.n)
    
    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        last_obs = [torch.tensor(ob, dtype=torch.float32, device=self.device) for ob in last_obs]
        obs = [torch.tensor(ob, dtype=torch.float32, device=self.device) for ob in obs]
        rewards = torch.tensor(np.array([rewards]), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.replay_buffer.store_experience(last_obs, actions, rewards, obs, done)

    def update_target_network(self):
        pass

    def train(self):
        obs, act, rew, next_obs, done = self.replay_buffer.sample_experience()
        if any([not alpha.is_frozen for alpha in self.n_alpha]):
            alpha_loss = self._compute_alpha_loss(obs, act, rew, next_obs, done)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha_loss = 0

        network_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
        self.network_optim.zero_grad()
        network_loss.backward()
        self.network_optim.step()

        for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
        return np.array([alpha_loss.cpu().detach().numpy(), network_loss.cpu().detach().numpy()])
    
    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            q_target_next = self.network_target(next_obs)
            q_target = rew + self.gamma * torch.max(q_target_next, dim=1, keepdim=True)[0] * (~done)
        q_expected = self.network_local(obs).gather(1, act.long())
        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss

    def _compute_alpha_loss(self, obs, act, rew, next_obs, done):
        critic_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
        alpha_loss = critic_loss
        for alpha in self.n_alpha:
            ent = alpha.get_entropy()
            alpha_loss += self.beta * ent
        return alpha_loss

    def hard_threshold_and_freeze_alpha(self):
        self.n_alpha[0].hard_threshold_and_freeze_alpha(2)
        self.n_alpha[1].hard_threshold_and_freeze_alpha(1)
        self.n_alpha[2].hard_threshold_and_freeze_alpha(1)

    def get_alpha_desc(self):
        desc = 'alpha of inter {}: '.format(self.inter.inter_idx)
        for alpha in self.n_alpha:
            desc += '\n{}'.format(alpha.get_desc())
        return desc
    
    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save({
            'network_local': self.network_local.state_dict(),
            'network_target': self.network_target.state_dict(),
            'n_alpha': self.n_alpha.state_dict(),
            'n_is_frozen': [alpha.is_frozen for alpha in self.n_alpha],
            'n_alive_idx': [alpha.get_alive_idx() for alpha in self.n_alpha],
        }, model_name)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        model_dict = torch.load(model_name)
        self.network_local.load_state_dict(model_dict['network_local'])
        self.network_target.load_state_dict(model_dict['network_target'])
        self.n_alpha.load_state_dict(model_dict['n_alpha'])
        for alpha_idx, alpha in enumerate(self.n_alpha):
            alpha.is_frozen = model_dict['n_is_frozen'][alpha_idx]
            alpha.n_alive_idx_after_frozen = model_dict['n_alive_idx'][alpha_idx]


class _Network(torch.nn.Module):
    def __init__(self, list_n_layer_dim, n_alpha, num_phase):
        super(_Network, self).__init__()
        assert len(list_n_layer_dim) == 3
        self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim = list_n_layer_dim
        self.n_alpha = n_alpha
        self.num_phase = num_phase

        self.input_2_first_layer = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.n_input_feature_dim[idx_input], self.n_layer_1_dim[idx_layer_1]),
                    torch.nn.ReLU(),
                )
                for idx_layer_1 in range(len(self.n_layer_1_dim))
            ])
            for idx_input in range(len(self.n_input_feature_dim))
        ])

        self.first_layer_2_second_layer = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.n_layer_1_dim[idx_layer_1], self.n_layer_2_dim[idx_layer_2]),
                    torch.nn.ReLU(),
                )
                for idx_layer_2 in range(len(self.n_layer_2_dim))
            ])
            for idx_layer_1 in range(len(self.n_layer_1_dim))
        ])

        self.second_layer_2_last_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.n_layer_2_dim[idx_layer_2], self.num_phase)
            for idx_layer_2 in range(len(self.n_layer_2_dim))
        ])

    def forward(self, obs): # [obs1(batch, feature), obs2, obs3]
        res_first_layer = [None for _ in range(len(self.n_layer_1_dim))]
        res_second_layer = [None for _ in range(len(self.n_layer_2_dim))]
        res_last_layer = None

        if not self.n_alpha[0].is_frozen:
            # BEFORE frozen: paths are weighted by the alpha term
            norm_alpha_0 = self.n_alpha[0].get_softmax_value()
            for idx_feature in range(len(self.n_input_feature_dim)):
                for jdx_layer_1 in range(len(self.n_layer_1_dim)):
                    elem = self.input_2_first_layer[idx_feature][jdx_layer_1](obs[idx_feature]) * norm_alpha_0[idx_feature]
                    if res_first_layer[jdx_layer_1] is None:
                        res_first_layer[jdx_layer_1] = elem
                    else:
                        res_first_layer[jdx_layer_1] += elem

            norm_alpha_1 = self.n_alpha[1].get_softmax_value()
            for idx_layer_1 in range(len(self.n_layer_1_dim)):
                for jdx_layer_2 in range(len(self.n_layer_2_dim)):
                    elem = self.first_layer_2_second_layer[idx_layer_1][jdx_layer_2](res_first_layer[idx_layer_1]) * norm_alpha_1[idx_layer_1]
                    if res_second_layer[jdx_layer_2] is None:
                        res_second_layer[jdx_layer_2] = elem
                    else:
                        res_second_layer[jdx_layer_2] += elem

            norm_alpha_2 = self.n_alpha[2].get_softmax_value()
            for idx_layer_2 in range(len(self.n_layer_2_dim)):
                elem = self.second_layer_2_last_layer[idx_layer_2](res_second_layer[idx_layer_2]) * norm_alpha_2[idx_layer_2]
                if res_last_layer is None:
                    res_last_layer = elem
                else:
                    res_last_layer += elem
        else:
            # AFTER frozen: only alive paths are activated
            n_alive_alpha_0 = self.n_alpha[0].get_alive_idx()
            n_alive_alpha_1 = self.n_alpha[1].get_alive_idx()
            n_alive_alpha_2 = self.n_alpha[2].get_alive_idx()

            for idx_feature in n_alive_alpha_0:
                for jdx_layer_1 in n_alive_alpha_1:
                    elem = self.input_2_first_layer[idx_feature][jdx_layer_1](obs[idx_feature])
                    if res_first_layer[jdx_layer_1] is None:
                        res_first_layer[jdx_layer_1] = elem
                    else:
                        res_first_layer[jdx_layer_1] += elem

            for idx_layer_1 in n_alive_alpha_1:
                for jdx_layer_2 in n_alive_alpha_2:
                    elem = self.first_layer_2_second_layer[idx_layer_1][jdx_layer_2](res_first_layer[idx_layer_1])
                    if res_second_layer[jdx_layer_2] is None:
                        res_second_layer[jdx_layer_2] = elem
                    else:
                        res_second_layer[jdx_layer_2] += elem

            for idx_layer_2 in n_alive_alpha_2:
                elem = self.second_layer_2_last_layer[idx_layer_2](res_second_layer[idx_layer_2])
                if res_last_layer is None:
                    res_last_layer = elem
                else:
                    res_last_layer += elem
        return res_last_layer


class Alpha(torch.nn.Module):
    EPS = 1e-9

    def __init__(self, elem_size, config):
        super(Alpha, self).__init__()
        self.elem_size = elem_size
        self.config = config
        self.alpha = torch.nn.Parameter(torch.ones(size=[self.elem_size]))
        self.is_frozen = False
        self.n_alive_idx_after_frozen = None  # only applicable after frozen

    def get_softmax_value(self):
        return F.softmax(self.alpha, dim=0)

    def get_alive_idx(self):
        return self.n_alive_idx_after_frozen

    def get_desc(self):
        return '{}, ent: {}'.format(
            '\t'.join(['{:.3f}'.format(elem) if elem > self.EPS else '-----' for elem in self.get_softmax_value().tolist()]),
            self.get_entropy()
        )

    def get_entropy(self):
        prob = self.get_softmax_value()
        ent = torch.sum(-prob * torch.log(prob))
        return ent

    def hard_threshold_and_freeze_alpha(self, num_alive_elem):
        self.is_frozen = True
        _, self.n_alive_idx_after_frozen = torch.topk(self.alpha, num_alive_elem)
        self.alpha.detach_()
        for idx in range(self.elem_size):
            if idx not in self.n_alive_idx_after_frozen:
                self.alpha[idx] = torch.tensor([self.EPS])
            else:
                self.alpha[idx] = torch.tensor([100.0])
        return self.alpha

def copy_model_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.clone())

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_size = obs_size 
        self.device = device

        self.memory = {
            'obs': self._get_obs_placeholder(),
            'act': np.empty((self.buffer_size, 1), dtype=np.int64),
            'rew': np.empty((self.buffer_size, 1), dtype=np.float32),
            'next_obs': self._get_obs_placeholder(),
            'done': np.empty((self.buffer_size, 1), dtype=np.bool_)
        }
        self._cur_idx = 0
        self.current_size = 0

    def _get_obs_placeholder(self):
        if isinstance(self.obs_size, list):
            return [np.empty((self.buffer_size, *siz), dtype=np.float32) for siz in self.obs_size]
        else:
            return np.empty((self.buffer_size, *self.obs_size), dtype=np.float32)

    def dump(self):
        return {
            "memory": self.memory,
            "_cur_idx": self._cur_idx,
            "current_size": self.current_size
        }

    def load(self, obj):
        self.memory = obj["memory"]
        self._cur_idx = obj["_cur_idx"]
        self.current_size = obj["current_size"]

    def reset(self):
        self._cur_idx = 0
        self.current_size = 0

    def store_experience(self, obs, act, rew, next_obs, done):
        if isinstance(self.obs_size, list):
            for feature_idx, ith_obs in enumerate(obs):
                self.memory['obs'][feature_idx][self._cur_idx] = ith_obs.cpu()
            for feature_idx, ith_next_obs in enumerate(next_obs):
                self.memory['next_obs'][feature_idx][self._cur_idx] = ith_next_obs.cpu()
        else:
            self.memory['obs'][self._cur_idx] = obs.cpu()
            self.memory['next_obs'][self._cur_idx] = next_obs.cpu()

        self.memory['act'][self._cur_idx] = act
        self.memory['rew'][self._cur_idx] = rew.cpu()
        self.memory['done'][self._cur_idx] = done

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self._cur_idx = (self._cur_idx + 1) % self.buffer_size

    def sample_experience(self, batch_size=None, idxs=None):
        batch_size = batch_size or self.batch_size
        if idxs is None:
            idxs = np.random.choice(self.current_size, batch_size, replace=True)

        if isinstance(self.obs_size, list):
            obs, next_obs = [], []
            for obs_feature_idx in range(len(self.obs_size)):
                obs.append(self._to_torch(self.memory['obs'][obs_feature_idx][idxs]))
                next_obs.append(self._to_torch(self.memory['next_obs'][obs_feature_idx][idxs]))
        else:
            obs = self._to_torch(self.memory['obs'][idxs])
            next_obs = self._to_torch(self.memory['next_obs'][idxs])

        act = self._to_torch(self.memory['act'][idxs])
        rew = self._to_torch(self.memory['rew'][idxs])
        done = self._to_torch(self.memory['done'][idxs])
        return obs, act, rew, next_obs, done

    def get_sample_indexes(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return np.random.choice(self.current_size, batch_size, replace=True)

    def _to_torch(self, np_elem):
        return torch.from_numpy(np_elem).to(self.device)

    def __str__(self):
        return str("current size: {} / {}".format(self.current_size, self.buffer_size))
