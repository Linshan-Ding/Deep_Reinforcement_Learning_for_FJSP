import random


class Config():
    """对象来保存代理/游戏的配置要求"""
    def __init__(self):
        self.seed = random.randint(1, 100)  # 随机种子
        self.environment_test = None  # 环境名称
        self.requirements_to_solve_game = None  # 配置需求
        self.num_episodes_to_run = None  # 运行周期
        self.file_to_save_data_results = None  # 保存结果数据的位置
        self.file_to_save_results_graph = None  # 保存结果图的位置
        self.use_GPU = None  # 是否使用GPU
        self.overwrite_existing_results_file = None  # 是否覆盖存在的结果文件
        self.save_model = True  # 是否保存模型
        self.hyper_parameters = self.hyper_parameter()  # 算法超参数

    def hyper_parameter(self):
        """算法超参数"""
        parameters = {
            "DA3C": {
                "learning_rate": 0.0003,
                "discount_rate": 0.99,
                "num_episodes_to_run": 1200,
                "gradient_clipping_norm": 1.0,
                "clip_rewards": True,
                "normalise_rewards": True,
                "epsilon_decay_rate_denominator": 1.0,
                "exploration_worker_difference": 2.0
            },
            "MP_PPO": {
                "actor_number": 5,
                "policy_update_round": 10,
                "num_episodes_to_run": 1000,
                "tau": 0.005,
                "learning_rate": 0.0003,
                "discount_rate": 0.99,
                "buffer_size": 10000,
                "batch_size": 256,
                "episodes_per_learning_round": 10,
                "learning_iterations_per_round": 10,
                "learning_iterations_per_round_actor": 10,
                "learning_iterations_per_round_critic": 10,
                "clip_epsilon": 0.2,
                "mu": 0,
                "theta": 0.15,
                "sigma": 0.2,
                "epsilon_decay_rate_denominator": 10,
                "clip_rewards": False,
                "normalized_rewards": True,
                "standardized_rewards": True,
                "gradient_clipping_norm": 1.0
            },
            "HMP_SAC": {
                "num_episodes_to_run": 2000,
                "learning_rate": 0.0003,
                "discount_rate": 0.99,
                "buffer_size": 10000,
                "batch_size": 256,
                "gradient_clipping_norm": 1.0,
                "min_steps_before_learning": 10000,
                "tau": 0.005,
                "learning_updates_per_learning_session": 10,
                "update_every_n_steps": 1000,
                "add_extra_noise": False,
                "do_evaluation_iterations": False,
                "entropy_term_weight": 0,
                "normalized_rewards": True,
                "standardized_rewards": True,
                "automatically_tune_entropy_hyper_parameter": True
            },
            "DDQN": {
                "num_episodes_to_run": 1000,
                "gradient_clipping_norm": 5.0,
                "tau": 0.005,
                "buffer_size": 100000,
                "batch_size": 1280,
                "learning_iterations": 1,
                "update_every_n_steps": 10,
                "epsilon_decay_rate_denominator": 10,
                "learning_rate": 0.000001,
                "discount_rate": 1
            },
            "MPTD3": {
                "learning_rate": 0.0003,
                "discount_rate": 0.99,
                "gradient_clipping_norm": 5.0,
                "tau": 0.005,
                "actor_number": 5,
                "action_size": 1,
                "state_size": 24,
                "num_episodes_to_run": 2000,
                "buffer_size": 10000,
                "batch_size": 256,
                "buffer_size_critic": 50000,
                "batch_size_critic": 1280,
                "mu": 0,
                "theta": 0.15,
                "sigma": 0.2,
                "action_noise_std": 0.2,
                "action_noise_clipping_range": 0.5,
                "update_every_n_steps": 5,
                "learning_updates_per_learning_session": 5,
                "delay_update": 20
            }
        }

        return parameters



