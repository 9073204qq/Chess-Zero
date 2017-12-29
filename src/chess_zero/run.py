import os
import sys
# noinspection PyPep8Naming
from keras import backend as K

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
	sys.path.append(_PATH_)


if __name__ == "__main__":
	K.set_epsilon(1e-05)
	K.set_floatx('float16')
	sys.setrecursionlimit(10000)
	from chess_zero import manager
	manager.start()
