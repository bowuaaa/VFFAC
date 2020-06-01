REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_msg_agent import RnnMsgAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY['rnn_msg'] = RnnMsgAgent