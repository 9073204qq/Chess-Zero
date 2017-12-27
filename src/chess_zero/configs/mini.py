class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 50
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 150
        self.play_config.c_puct = 1 # lower  = prefer mean action value
        self.play_config.tau_decay_rate = 0.6 # I need a better distribution...
        self.play_config.virtual_loss = 2
        self.evaluate_latest_first = True
        self.max_game_length = 1000


class PlayDataConfig:
    def __init__(self):
        self.min_elo_policy =  400 # 0 weight
        self.max_elo_policy =  1900 # 1 weight
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 50
        self.max_file_num = 150


class PlayConfig:
    def __init__(self):
        self.max_processes = 3
        self.search_threads = 16
        self.vram_frac = 1.0
        self.simulation_num_per_move = 200
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.99
        self.virtual_loss = 2
        self.resign_threshold = -0.8
        self.min_resign_turn = 5
        self.max_game_length = 1000


class TrainerConfig:
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.types_allowed = ["standard", "blitz", "lightning", 'wild/2','wild/3','wild/4','wild/5','wild/8','wild/8a','wild/fr']
        self.replace_rate = 1.0
        self.cleaning_processes = 8 # RAM explosion...
        self.vram_frac = 1.0
        self.batch_size = 256 # tune this to your gpu memory
        self.epoch_to_checkpoint = 1
        self.dataset_size = 50000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.loss_weights = [1.25, 1.0] # [policy, value] I'm using policy weights so scale back up


class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 3
    cnn_filter_size = 3
    res_layer_num = 12
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = False
    input_depth = 18
