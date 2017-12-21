import enum
import chess.pgn
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# input planes
pieces_order = 'KQRBNPkqrbnp'
castling_order = 'KQkq'
# fifty-move-rule
# en passant

ind = {pieces_order[i]: i for i in range(12)}

class ChessEnv:

    def __init__(self):
        self.board = None
        self.num_halfmoves = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = chess.Board(board)
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    @property
    def whitewon(self):
        return self.winner == Winner.white

    def step(self, action: str, check_over = True):
        """
        :param int|None action, None is resign
        :return:
        """
        if check_over and action is None:
            self._resigned()
            return self.board, {}

        self.board.push_uci(action)

        self.num_halfmoves += 1

        if check_over and self.board.result(claim_draw=True) != "*":
            self._game_over()

        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            result = self.board.result(claim_draw = True)
            if result == '1-0':
                self.winner = Winner.white
            elif result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def _resigned(self):
        self._win_another_player()
        self._game_over()
        self.resigned = True

    def _win_another_player(self):
        if self.board.turn == chess.BLACK:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def adjudicate(self):
        self.resigned = False
        self.done = True
        score = self.testeval(absolute = True)
        if abs(score) < 0.01:
            self.winner= Winner.draw
        elif score > 0:
            self.winner = Winner.white
        else:
            self.winner = Winner.black

    def ending_average_game(self):
        self.resigned = False
        self.done = True
        self.winner = Winner.draw

    def canonical_input_planes(self):
        return canon_input_planes(self.board.fen())

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def replace_tags(self):
        return replace_tags_board(self.board.fen())

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    def deltamove(self, fen_next):
        moves = list(self.board.legal_moves)
        for mov in moves:
            self.board.push(mov)
            fee = self.board.fen()
            self.board.pop()
            if fee == fen_next:
                return mov.uci()
        return None

    def testeval(self, absolute=False) -> float:
        return testeval(self.board.fen(), absolute)

def testeval(fen, absolute = False) -> float:
    piecevals = {'K': 3, 'Q': 9, 'R': 5,'B': 3.25,'N': 3,'P': 1} # K is always on board....
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue
        #assert c.upper() in piecevals   
        if c.isupper():
            ans += piecevals[c]
            tot += piecevals[c]
        else:
            ans -= piecevals[c.upper()]
            tot += piecevals[c.upper()]
    v = ans/tot
    if not absolute and isblackturn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3) # arbitrary

def check_current_planes(realfen, planes):
    cur = planes[0:12]
    assert cur.shape == (12, 8, 8)
    fakefen = ["1"] * 64
    for i in range(12):
        for rank in range(8):
            for file in range(8):
                if cur[i][rank][file] == 1:
                    assert fakefen[rank * 8 + file] == '1'
                    fakefen[rank * 8 + file] = pieces_order[i]

    castling = planes[12:16]
    fiftymove = planes[16][0][0]
    ep = planes[17]

    castlingstring = ""
    for i in range(4):
        if castling[i][0][0] == 1:
            castlingstring += castling_order[i]

    if len(castlingstring) == 0:
        castlingstring = '-'

    epstr = "-"
    for rank in range(8):
        for file in range(8):
            if ep[rank][file] == 1:
                epstr = coord_to_alg((rank, file))

    realfen = maybe_flip_fen(realfen, flip=isblackturn(realfen))
    realparts = realfen.split(' ')
    assert realparts[1] == 'w'
    assert realparts[2] == castlingstring
    assert realparts[3] == epstr
    assert int(realparts[4]) == fiftymove
    # realparts[5] is the fifty-move clock, discard that
    return "".join(fakefen) == replace_tags_board(realfen)

def canon_input_planes(fen):
    fen = maybe_flip_fen(fen, isblackturn(fen))
    return all_input_planes(fen)

def all_input_planes(fen):
    current_aux_planes = aux_planes(fen)

    history_both = to_planes(fen)

    ret = np.vstack((history_both, current_aux_planes))
    assert ret.shape == (18, 8, 8)
    return ret

def maybe_flip_fen(fen, flip = False):
    if flip == False:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    return "/".join( [swapall(row) for row in reversed(rows)] ) \
        + " " + ('w' if foo[1]=='b' else 'b') \
        + " " + "".join( sorted( swapall(foo[2]) ) ) \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]

def aux_planes(fen):
    foo = fen.split(' ')

    en_passant = np.zeros((8, 8), dtype=np.float32)
    eps = alg_to_coord(foo[3])
    if eps != '-':
        en_passant[eps[0]][eps[1]] = 1

    fifty_move_count = int(foo[4])
    fifty_move = np.full((8,8), fifty_move_count, dtype=np.float32)

    castling = foo[2]
    auxiliary_planes = [np.full((8,8), int('K' in castling), dtype=np.float32), \
                        np.full((8,8), int('Q' in castling), dtype=np.float32), \
                        np.full((8,8), int('k' in castling), dtype=np.float32), \
                        np.full((8,8), int('q' in castling), dtype=np.float32), \
                        fifty_move, \
                        en_passant]

    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (6,8,8)
    return ret

# FEN board is like this:
# a8 b8 .. h8
# a7 b7 .. h7
# .. .. .. ..
# a1 b1 .. h1
# 
#  0  1 ..  7
#  8  9 .. 15
# .. .. .. ..
# 56 57 .. 63

# my planes are like this:
# 00 01 .. 07
# 10 11 .. 17
# .. .. .. ..
# 70 71 .. 77
#

def alg_to_coord(alg):
    rank = 8 - int(alg[1])        # 0-7
    file = ord(alg[0]) - ord('a') # 0-7
    return (rank, file)

def coord_to_alg(coord):
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number

def to_planes(fen):
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape = (12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8)
    return pieces_both

def replace_tags_board(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")

def isblackturn(fen):
    return fen.split(" ")[1] == 'b'