from common.registry import Registry
@Registry.register_model('base')
class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.
    '''
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.sub_agents = 1
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.ob_list = Registry.mapping['model_mapping']['setting'].param['ob_list']
        self.reward_list = Registry.mapping['model_mapping']['setting'].param['reward_list']
        
    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()

    def get_action_prob(self, ob, phase):
        return None
