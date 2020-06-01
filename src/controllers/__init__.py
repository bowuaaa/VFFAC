REGISTRY = {}

from .basic_controller import BasicMAC
from .vffac_controller import VffacMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['vffac_mac'] = VffacMAC