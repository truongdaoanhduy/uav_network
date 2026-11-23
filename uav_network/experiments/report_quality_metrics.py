"""Terminal ground-truth *scoring only* — never used as policy observation.

Counts refer to distinct cell report hypotheses and real targets matched to
these cells. An unknown report's true/false status may be inspected only after
simulator terminal scoring, never before network delivery or action selection.
"""
from __future__ import annotations


def terminal_report_quality(record):
    """Return counted precision for local confirmations and delivered reports.

    Do not report 100% when no reports have been confirmed/delivered.
    Omit precision entirely for the legacy target-ID format (no false counts).
    """
    if 'false_confirmed_cells' not in record or 'false_reports_delivered' not in record:
        return {}
    found=int(record['targets_found']);delivered=int(record['targets_delivered'])
    false_local=int(record['false_confirmed_cells'])
    false_gcs=int(record['false_reports_delivered'])
    for k,v in (('targets_found',found),('targets_delivered',delivered),
                ('false_confirmed_cells',false_local),('false_reports_delivered',false_gcs)):
        if v<0 or float(record[k])!=v:
            raise ValueError(f'Invalid {k} counter')
    if delivered>found:
        raise ValueError('True deliveries cannot exceed confirmed targets')
    quality={'false_confirmed_cells':false_local,'false_reports_delivered':false_gcs}
    if found+false_local>0:
        quality['confirmation_precision_percent']=100.*found/(found+false_local)
    if delivered+false_gcs>0:
        quality['gcs_report_precision_percent']=100.*delivered/(delivered+false_gcs)
    return quality
