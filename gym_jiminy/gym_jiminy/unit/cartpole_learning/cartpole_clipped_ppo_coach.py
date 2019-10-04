from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule, ScheduleParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(4000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = ClippedPPOAgentParameters()

# Agent params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1024)

# NN configuration
agent_params.network_wrappers['main'].input_embedders_parameters = {
    'observation': InputEmbedderParameters(scheme=[])
}
agent_params.network_wrappers['main'].learning_rate = 0.001

################
#  Environment #
################
env_params = GymVectorEnvironment(level='gym_jiminy.envs.cartpole:JiminyCartPoleEnv')

################
#   Learning   #
################
graph_manager = BasicRLGraphManager(agent_params=agent_params,
                                    env_params=env_params,
                                    schedule_params=SimpleSchedule())
graph_manager.improve()