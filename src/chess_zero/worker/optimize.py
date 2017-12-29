import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging import getLogger
from random import shuffle

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, SGD
import chess

from chess_zero.agent.model_chess import ChessModel
from chess_zero.config import Config
from chess_zero.env.chess_env import canon_input_planes, is_black_turn, testeval, canon_fen
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from chess_zero.lib.model_helper import load_best_model_weight
from chess_zero.worker.sl import get_buffer, get_games_from_all_files

logger = getLogger(__name__)
precision = np.float16

def start(config: Config):
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ChessModel
        self.loaded_filenames = set()
        self.dataset_size = self.config.trainer.dataset_size
        self.dataset = deque(maxlen=self.dataset_size), deque(maxlen=self.dataset_size), deque(maxlen=self.dataset_size)

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        # self.filenames = deque(get_game_data_filenames(self.config.resource))
        # shuffle(self.filenames)
        total_steps = self.config.trainer.start_total_steps

        while True:
            self.more_data = True
            self.games = self.get_all_games()
            self.fill_queue2(frac = 1.0-self.config.trainer.replace_rate)
            while True:
                self.fill_queue2(frac = self.config.trainer.replace_rate)
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                self.save_current_model()
                if not self.more_data:
                    break

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        #tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02)
                             #callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        #opt = SGD(lr=0.1,momentum=0.9)
        opt = Adam(epsilon=1e-5)
        losses = [cross_entropy_with_logits, 'mean_squared_error'] # avoid overfit for supervised 
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def fill_queue2(self, frac):
        a, b, c = self.dataset
        amt = frac * self.dataset_size
        replaced = 0
        while replaced < amt:
            try:
                x, y, z = next(self.games)
                a.extend(x)
                b.extend(y)
                c.extend(z)
                replaced += len(x)
            except StopIteration:
                a.popleft()
                b.popleft()
                c.popleft()
                replaced += 1
        print(f"replaced {replaced} datapoints")


    def fill_queue(self):
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                logger.debug(f"loading data from {filename}")
                futures.append(executor.submit(load_data_from_file,filename))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                for x,y in zip(self.dataset,futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file,filename))

    def collect_all_loaded_data(self):
        state_ary,policy_ary,value_ary=self.dataset

        state_ary1 = np.asarray(state_ary, dtype=precision)
        policy_ary1 = np.asarray(policy_ary, dtype=precision)
        value_ary1 = np.asarray(value_ary, dtype=precision)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        model = ChessModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model


    def get_all_games(self):
        # noinspection PyAttributeOutsideInit
        with ProcessPoolExecutor(max_workers=7) as executor:
            games = get_games_from_all_files(self.config)
            shuffle(games)
            for res in as_completed([executor.submit(load_data_from_game, self.config, game) for game in games]): #poisoned reference (memleak)
                yield res.result()
        self.more_data = False

def cross_entropy_with_logits(y_true, y_pred_logits):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred_logits)
    return tf.reduce_mean(tf.reduce_sum(y_true,axis=1)*tf.reduce_logsumexp(y_pred_logits,axis=1)
     - tf.reduce_sum(y_true*y_pred_logits,axis=1))

def load_data_from_game(config, game):
    buf = get_buffer(config,game)
    return convert_to_cheating_data(buf)


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)


def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:
        #print(state_fen, value)

        state_planes = canon_input_planes(state_fen, check=False)

        if is_black_turn(state_fen):
            policy = Config.flip_policy(policy)
        
        # try:
        #     my_move = chess.Move.from_uci(Config.labels[np.argmax(policy)])
        #     canon_fe = canon_fen(state_fen)
        #     my_legal = list(chess.Board(canon_fe).legal_moves)
        #     assert my_move in my_legal
        # except AssertionError:
        #     print(my_move, canon_fe)

        move_number = int(state_fen.split(' ')[5])
        value_certainty = 0#min(move_number,10)/10 # this NN really sucks at material evaluation
        sl_value = value * value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=precision), np.asarray(policy_list, dtype=precision), np.asarray(value_list, dtype=precision)
