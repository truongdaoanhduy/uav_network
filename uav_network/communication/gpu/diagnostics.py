"""Fixed-size passive device telemetry; one thread owns each world row."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from ..diagnostics import COUNTERS, STATS

COUNTER_COUNT = wp.constant(len(COUNTERS))
STAT_COUNT = wp.constant(len(STATS))

DATA_ATTEMPTS = wp.constant(0)
DATA_RETRIES = wp.constant(1)
DATA_PHY_SUCCESSES = wp.constant(2)
DATA_PHY_FAILURES = wp.constant(3)
ACK_ATTEMPTS = wp.constant(4)
ACK_PHY_SUCCESSES = wp.constant(5)
ACK_PHY_FAILURES = wp.constant(6)
DATA_DECODE_COUNT = wp.constant(7)
DATA_WEAK_SIGNAL_FAILURES = wp.constant(8)
DATA_INTERFERENCE_FAILURES = wp.constant(9)
DATA_OVERLAP_DECODES = wp.constant(10)
DATA_COLLISION_FAILURES = wp.constant(11)
DATA_LOS_DECODES = wp.constant(12)
DATA_LOS_SUCCESSES = wp.constant(13)
DATA_NLOS_DECODES = wp.constant(14)
DATA_NLOS_SUCCESSES = wp.constant(15)
DATA_PEER_DECODES = wp.constant(16)
DATA_PEER_SUCCESSES = wp.constant(17)
DATA_GCS_DECODES = wp.constant(18)
DATA_GCS_SUCCESSES = wp.constant(19)
QUEUE_FULL = wp.constant(20)
PACKET_LIFETIME = wp.constant(21)
RETRY_LIMIT = wp.constant(22)
ENERGY_DENIALS = wp.constant(23)
CCA_CHECKS = wp.constant(24)
CCA_BUSY = wp.constant(25)
BACKOFF_DRAWS = wp.constant(26)
BACKOFF_FREEZES = wp.constant(27)
CUSTODY_PEER_TRANSFERS = wp.constant(28)
DIRECT_GCS_REPORTS = wp.constant(29)
RELAYED_GCS_REPORTS = wp.constant(30)
ACTION_SLOTS = wp.constant(31)
ACTION_ACTIVE = wp.constant(32)
ACTION_REPORT_BEARING = wp.constant(33)
ACTION_RECEIVER_ELIGIBLE = wp.constant(34)
ACTION_SILENT = wp.constant(35)
ACTION_NO_OPPORTUNITY = wp.constant(36)
DATA_ENERGY_DENIALS = wp.constant(37)
ACK_ENERGY_DENIALS = wp.constant(38)
ACK_TIMEOUTS = wp.constant(39)
DATA_BELOW_REQUIRED_POWER_ATTEMPTS = wp.constant(40)
RELAY_TOWARD_GCS_TRANSFERS = wp.constant(41)
RELAY_AWAY_FROM_GCS_TRANSFERS = wp.constant(42)
STAT_DATA_TX_POWER_W = wp.constant(0)
STAT_ACK_TX_POWER_W = wp.constant(1)
STAT_DATA_PEER_TX_POWER_W = wp.constant(2)
STAT_DATA_GCS_TX_POWER_W = wp.constant(3)
STAT_DATA_SNR_DB = wp.constant(4)
STAT_DATA_SINR_DB = wp.constant(5)
STAT_DATA_RSSI_DBM = wp.constant(6)
STAT_DATA_DISTANCE_M = wp.constant(7)
STAT_DATA_REQUIRED_POWER_W = wp.constant(8)
STAT_DATA_POWER_MARGIN_DB = wp.constant(9)
STAT_RELAY_GCS_DISTANCE_IMPROVEMENT_M = wp.constant(10)


@wp.func
def reset(s: WorldState, e: int):
    for j in range(COUNTER_COUNT):
        s.diagnostic_counts[e,j] = wp.int64(0)
    for j in range(STAT_COUNT):
        for k in range(4):
            s.diagnostic_stats[e,j,k] = wp.float64(0)
    for j in range(8):
        s.delivery_hop_histogram[e,j] = wp.int64(0)
    for j in range(2):
        for k in range(6):
            s.power_histogram[e,j,k] = wp.int64(0)

@wp.func
def sample(s: WorldState, e: int, index: int, value: wp.float64):
    if s.diagnostic_stats[e,index,0] == wp.float64(0):
        s.diagnostic_stats[e,index,2] = value
        s.diagnostic_stats[e,index,3] = value
    else:
        s.diagnostic_stats[e,index,2] = wp.min(s.diagnostic_stats[e,index,2],value)
        s.diagnostic_stats[e,index,3] = wp.max(s.diagnostic_stats[e,index,3],value)
    s.diagnostic_stats[e,index,0] += wp.float64(1)
    s.diagnostic_stats[e,index,1] += value


@wp.func
def emission(s: WorldState, p: WorldParams, e: int, packet: int, destination: int,
             kind: int, power: wp.float64):
    stat = STAT_DATA_TX_POWER_W
    count = DATA_ATTEMPTS
    if kind == 1:
        stat = STAT_ACK_TX_POWER_W
        count = ACK_ATTEMPTS
    s.diagnostic_counts[e,count] += wp.int64(1)
    sample(s,e,stat,power)
    bin_index = int(0)
    if power >= wp.float64(0.1):
        bin_index += 1
    if power >= wp.float64(0.2):
        bin_index += 1
    if power >= wp.float64(0.3):
        bin_index += 1
    if power >= wp.float64(0.4):
        bin_index += 1
    if power >= wp.float64(0.5):
        bin_index += 1
    s.power_histogram[e,kind,bin_index] += wp.int64(1)
    if kind == 0:
        if s.pkt_attempt[e,packet] > 1:
            s.diagnostic_counts[e,DATA_RETRIES] += wp.int64(1)
        link_stat = STAT_DATA_PEER_TX_POWER_W
        if destination == p.n:
            link_stat = STAT_DATA_GCS_TX_POWER_W
        sample(s,e,link_stat,power)


@wp.func
def decode(s: WorldState, p: WorldParams, e: int, kind: int, destination: int,
           snr: wp.float64, sinr: wp.float64, rssi: wp.float64, distance: wp.float64,
           line_of_sight: bool, collided: bool, success: bool):
    result = DATA_PHY_FAILURES
    if kind == 1:
        result = ACK_PHY_FAILURES
    if success:
        result -= 1
    s.diagnostic_counts[e,result] += wp.int64(1)
    if kind == 0:
        s.diagnostic_counts[e,DATA_DECODE_COUNT] += wp.int64(1)
        if collided:
            s.diagnostic_counts[e,DATA_OVERLAP_DECODES] += wp.int64(1)
            if not success:
                s.diagnostic_counts[e,DATA_COLLISION_FAILURES] += wp.int64(1)
        visibility = DATA_NLOS_DECODES
        if line_of_sight:
            visibility = DATA_LOS_DECODES
        s.diagnostic_counts[e,visibility] += wp.int64(1)
        link = DATA_PEER_DECODES
        if destination == p.n:
            link = DATA_GCS_DECODES
        s.diagnostic_counts[e,link] += wp.int64(1)
        if success:
            s.diagnostic_counts[e,visibility+1] += wp.int64(1)
            s.diagnostic_counts[e,link+1] += wp.int64(1)
        else:
            failure = DATA_INTERFERENCE_FAILURES
            if snr < p.sinr_threshold:
                failure = DATA_WEAK_SIGNAL_FAILURES
            s.diagnostic_counts[e,failure] += wp.int64(1)
        sample(s,e,STAT_DATA_SNR_DB,snr)
        sample(s,e,STAT_DATA_SINR_DB,sinr)
        sample(s,e,STAT_DATA_RSSI_DBM,rssi)
        sample(s,e,STAT_DATA_DISTANCE_M,distance)
