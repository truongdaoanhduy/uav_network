"""Original 18d paper-style TRAINING outcome keys, from real terminal evidence.

The algorithm never consumes this helper. Missing times/coverage AUC are
omitted rather than imputed. Every array is from one real completed episode.
"""
from __future__ import annotations

import math


def _capped_times(values, horizon):
    if not isinstance(values, (list, tuple)):
        return None
    capped=[]
    for value in values:
        if value is None:
            capped.append(horizon)
        else:
            t=float(value)
            if not math.isfinite(t) or t < 0 or t > horizon + 1e-5:
                raise ValueError("Observed target timestamp outside mission horizon")
            capped.append(min(t,horizon))
    return capped


def terminal_paper_metrics(record):
    """Ground-truth scoring here is logging-only, never an Actor observation."""
    result={
        'training/discovery_rate_percent':float(record['search_rate_percent']),
        'training/gcs_delivery_rate_percent':float(record['delivery_rate_percent']),
        'training/mission_success_fraction':float(bool(record['success'])),
        'training/mission_success_rate_percent':100.*float(bool(record['success'])),
        'training/search_delivery_gap_percentage_points':float(record['search_rate_percent'])-float(record['delivery_rate_percent']),
    }
    for key,label in [('targets_found','training/targets_found'),('targets_delivered','training/targets_delivered')]:
        if key in record:
            result[label]=int(record[key])
    horizon=record.get('episode_horizon_s')
    if horizon is not None:
        horizon=float(horizon)
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('Invalid mission horizon')
        result['training/mission_completion_time_capped_s']=(
            float(record['episode_length'])*float(record.get('episode_dt_s',1))
            if bool(record['success']) else horizon)
    details=record.get('report_diagnostics')
    if isinstance(details,dict) and horizon:
        found=_capped_times(details.get('first_discovery_s'),horizon)
        delivered=_capped_times(details.get('first_delivery_s'),horizon)
        if found is not None and delivered is not None:
            if not found or len(found)!=len(delivered):
                raise ValueError('Incomplete per-target timestamp arrays')
            if ('targets_found' in record and
                sum(v<horizon for v in found)>record['targets_found']):
                raise ValueError('Discovered timestamps exceed confirmed targets')
            # Exactly the held-out GPU formula: AUC = 1-mean(capped_time)/H.
            dtime=math.fsum(found)/len(found)
            gtime=math.fsum(delivered)/len(delivered)
            if any(g < f-1e-5 for f,g in zip(found,delivered)):
                raise ValueError('GCS delivery before target discovery')
            result.update({
                'training/discovery_auc':max(0.,min(1.,1.-dtime/horizon)),
                'training/delivery_auc':max(0.,min(1.,1.-gtime/horizon)),
                'training/gcs_delivery_auc':max(0.,min(1.,1.-gtime/horizon)),
                'training/discovery_time_capped_mean_s':dtime,
                'training/gcs_delivery_time_capped_mean_s':gtime})
    if record.get('coverage_percent') is not None:
        result['training/map_coverage_percent']=float(record['coverage_percent'])
    if record.get('coverage_auc') is not None:
        auc=float(record['coverage_auc'])
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError('Invalid observed coverage AUC')
        result['training/coverage_auc']=auc
    energy=record.get('energy') or {}
    network=record.get('network') or {}
    if 'propulsion_energy_j' in energy:
        joules=float(energy['propulsion_energy_j'])+float(network.get('uav_communication_energy_j',0))
        if not math.isfinite(joules) or joules<0:
            raise ValueError('Invalid energy')
        result['training/total_uav_energy_j']=joules
        if record.get('targets_delivered',0)>0:
            result['training/energy_per_delivered_target_j']=joules/int(record['targets_delivered'])
    if network.get('enabled'):
        sent=network.get('generated_packets');received=network.get('received_packets')
        if sent is not None and received is not None and int(sent)>0:
            if not 0 <= int(received) <= int(sent):
                raise ValueError('Invalid packet counters')
            result['training/packet_pdr_percent']=100.*int(received)/int(sent)
    return result
