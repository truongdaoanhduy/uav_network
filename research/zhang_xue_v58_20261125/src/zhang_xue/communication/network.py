"""Discrete-event CPU CSMA/CA radio network, composed from protocol stages."""
from .common import *
from .lifecycle import _NetworkLifecycleMixin
from .ledger import _NetworkLedgerMixin
from .channel import _NetworkChannelMixin
from .delivery import _NetworkDeliveryMixin
from .scheduler import _NetworkSchedulerMixin

class SevenRadioCsmaCaNetwork(_NetworkLifecycleMixin, _NetworkLedgerMixin, _NetworkChannelMixin, _NetworkDeliveryMixin, _NetworkSchedulerMixin):
    """Seven distinct radio identities, packet custody, ACK and route lifecycle."""
