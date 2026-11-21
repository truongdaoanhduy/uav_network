"""Source-provenanced correlated DS, cooperative SPRT, M2 GCS map protocol.

The only way a map changes is ingest(Event, arrived_at). In particular the
GCS cannot access any UAV map by shared memory. A transport must explicitly
call ingest after full packet reassembly. Ground truth is accepted ONLY by
MultiLayerMapMetrics and NEVER by an actor, critic or protocol event.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import numpy as np

from ..environment.correlated_mapping import Event, ObstacleGeometryEvent, SharedBiasLikelihood
from ..environment.ds_evidence import combine_evidence


def map_evidence_weight(p1: float, p0: float, positive: bool) -> float:
    if not all(math.isfinite(x) for x in (p1,p0)) or not (0 < p0 <= p1 < 1):
        raise ValueError("invalid conditional likelihoods")
    if positive:
        return max(0., min(1-1e-12, 1-p0/p1))
    return max(0., min(1-1e-12, 1-(1-p1)/(1-p0)))


def stable_mass_from_llr_totals(positive_llr: float, negative_llr: float):
    """Exact normalized DS combination of simple-support evidence in log space.

    LLR>0 supports occupied, LLR<0 supports empty. Storing cumulative support
    (rather than rounded masses) avoids K≈1 cancellation after long conflict.
    """
    if not (math.isfinite(positive_llr) and math.isfinite(negative_llr)):
        raise ValueError("non-finite accumulated evidence")
    if positive_llr < 0 or negative_llr < 0:
        raise ValueError("LLR support totals cannot be negative")
    # log(1-exp(-x)) is -inf at x=0 and stable for very small x.
    def log_one_minus_exp_neg(x):
        return math.log(-math.expm1(-x)) if x>0 else -math.inf
    log_f=log_one_minus_exp_neg(positive_llr)-negative_llr
    log_e=-positive_llr+log_one_minus_exp_neg(negative_llr)
    log_u=-positive_llr-negative_llr
    norm=float(np.logaddexp(np.logaddexp(log_f,log_e),log_u))
    return (math.exp(log_f-norm),math.exp(log_e-norm),math.exp(log_u-norm))


@dataclass(frozen=True)
class DecisionNotice:
    """Verified/rejected/retracted claim with monotonically increasing version.

    A cell is an ESTIMATED ground-plane cell, NOT a simulator target identity.
    The GCS keeps the claim history; a late older packet cannot undo a newer
    correction. For cooperative decisions each sender can submit a claim.
    """
    message_id: str
    source: int
    cell: int
    revision: int
    status: str
    tick: int

    def __post_init__(self):
        if not self.message_id or self.source < 0 or self.cell < 0 or self.revision < 1 or self.tick < 0:
            raise ValueError("invalid decision notice identity")
        if self.status not in ("pending", "verified", "rejected", "retracted"):
            raise ValueError("invalid decision status")


@dataclass
class CandidateState:
    cutoff_tick: int
    cutoff_event: str
    cutoff_key: tuple
    status: str = "pending"
    sprt_n: int = 0
    sprt_llr: float = 0.
    revision: int = 0


class CooperativeWorldMap:
    """Local perception map for one node (a UAV OR GCS). No network bypass."""
    def __init__(self, num_cells:int, ds_threshold:float=.95, alpha:float=.05, beta:float=.05):
        if num_cells < 1 or not (0 < ds_threshold <= 1):
            raise ValueError("invalid map dimensions/threshold")
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha+beta < 1):
            raise ValueError("SPRT false alarm/miss budgets must sum below one")
        self.num_cells=int(num_cells)
        self.ds_threshold=ds_threshold
        self.upper=math.log((1-beta)/alpha)
        self.lower=math.log(beta/(1-alpha))
        self.target_masses=np.zeros((num_cells,3),dtype=np.float64)
        self.obstacle_masses=np.zeros((num_cells,3),dtype=np.float64)
        self.target_masses[:,2]=self.obstacle_masses[:,2]=1
        self.coverage=np.zeros(num_cells,dtype=np.int32)
        self.last_observed=np.full(num_cells,-1,dtype=np.int64)
        self.last_arrival=np.full(num_cells,-1,dtype=np.int64)
        self.events={}
        self._by_cell_layer={}
        self.candidates={}
        self.decision_changes=[]
        self.link_contacts={}  # observed delivery/ACK only, not simulator link oracle
        self.claims={}  # (source,cell) -> latest DecisionNotice
        self.report_history=[]  # all accepted versions and their arrival times
        self.geometry_events={}  # immutable noisy 3D obstacle observations
        self.obstacle_tracks=[]  # geometry estimates, NOT simulator geometry IDs

    @property
    def events_seen(self):return len(self.events)

    def ingest(self,event:Event,*,arrived_at:int) -> bool:
        if not isinstance(event,Event) or not (0 <= event.cell < self.num_cells):
            raise ValueError("invalid observation event/cell")
        if arrived_at < 0:
            raise ValueError("arrival time must be nonnegative")
        old=self.events.get(event.event_id)
        if old is not None:
            if old != event:raise ValueError("conflicting content for an existing event ID")
            return False
        self.events[event.event_id]=event
        key=(event.cell,event.layer)
        self._by_cell_layer.setdefault(key,[]).append(event)
        self.coverage[event.cell]+=1
        self.last_arrival[event.cell]=max(self.last_arrival[event.cell],arrived_at)
        self.last_observed[event.cell]=max(self.last_observed[event.cell],event.tick)
        self._replay(event.cell,event.layer,arrived_at)
        return True

    def _replay(self,cell:int,layer:str,arrived_at:int):
        readings=sorted(self._by_cell_layer[(cell,layer)],key=lambda event:event.sort_key)
        evidence_engine=SharedBiasLikelihood()
        masses=(0.,0.,1.)
        positive_llr=0.
        negative_llr=0.
        candidate_cutoff=None
        sprt=0.
        n=0
        for event in readings:
            llr=evidence_engine.observe(event)
            if llr>0:
                positive_llr+=llr
            else:
                negative_llr-=llr
            masses=stable_mass_from_llr_totals(positive_llr,negative_llr)
            if layer=="target":
                if candidate_cutoff is None and masses[0] >= self.ds_threshold:
                    candidate_cutoff=event.sort_key
                elif candidate_cutoff is not None and event.sort_key > candidate_cutoff:
                    # Retain ALL prior events in the conditional latent likelihood,
                    # but none of the candidate-trigger events contribute to SPRT.
                    sprt+=llr
                    n+=1
        if layer=="target":
            self.target_masses[cell]=masses
            old=self.candidates.get(cell)
            if candidate_cutoff is None:
                if old is not None:
                    # Revisions can be triggered by out-of-order negative evidence.
                    self.decision_changes.append((cell,"retracted",arrived_at,old.revision+1))
                    self.candidates.pop(cell)
                return
            decision="verified" if sprt>=self.upper else ("rejected" if sprt<=self.lower else "pending")
            cutoff_event=candidate_cutoff[2]
            new=CandidateState(candidate_cutoff[0],cutoff_event,candidate_cutoff,
                               decision,n,sprt,old.revision if old else 0)
            if old is None or old.status!=decision or old.cutoff_key!=candidate_cutoff:
                new.revision=(old.revision if old else 0)+1
                self.decision_changes.append((cell,decision,arrived_at,new.revision))
            self.candidates[cell]=new
        else:
            self.obstacle_masses[cell]=masses

    def drain_decisions(self):
        changes=list(self.decision_changes)
        self.decision_changes.clear()
        return changes

    def make_decision_notices(self,*,source:int):
        results=[]
        for cell,status,tick,revision in self.drain_decisions():
            results.append(DecisionNotice(f"u{source}:c{cell}:v{revision}",
                                          source,cell,revision,status,tick))
        return results

    def ingest_notice(self,notice:DecisionNotice,*,arrived_at:int):
        if not isinstance(notice,DecisionNotice) or not 0 <= notice.cell < self.num_cells:
            raise ValueError("invalid decision notice")
        key=(notice.source,notice.cell)
        previous=self.claims.get(key)
        if previous is not None:
            if notice.revision < previous.revision:return False
            if notice.revision == previous.revision:
                if notice != previous:raise ValueError("conflicting decision revision")
                return False
        self.claims[key]=notice
        self.report_history.append((notice,arrived_at))
        return True

    def consensus_status(self,cell:int):
        """One GCS cell status, without ground-truth or privileged UAV state."""
        if not 0 <= cell < self.num_cells:
            raise IndexError(cell)
        local=self.candidates.get(cell)
        if local and local.status in ("verified","rejected"):
            return local.status  # fused receiver-local verification evidence
        source_states=[n.status for (src,c),n in self.claims.items() if c==cell]
        votes=set(x for x in source_states if x != "retracted")
        if "verified" in votes and "rejected" in votes:
            return "disputed"
        if "verified" in votes:return "verified"
        if "rejected" in votes:return "rejected"
        if "pending" in votes or (local and local.status=="pending"):
            return "pending"
        return "unknown"

    def ingest_geometry(self,event:ObstacleGeometryEvent,*,arrived_at:int):
        if not isinstance(event,ObstacleGeometryEvent):
            raise ValueError("obstacle geometry detection required")
        old=self.geometry_events.get(event.event_id)
        if old is not None:
            if old != event:raise ValueError("conflicting geometry event ID")
            return False
        self.geometry_events[event.event_id]=event
        self._rebuild_geometry_tracks()
        return True

    def _rebuild_geometry_tracks(self):
        tracks=[]
        # Deterministic replay despite arbitrarily delayed multi-hop packets.
        for ev in sorted(self.geometry_events.values(),key=lambda e:e.sort_key):
            eligible=[]
            for index,tr in enumerate(tracks):
                dist=math.hypot(ev.x-tr['x'],ev.y-tr['y'])
                gate=3.*math.hypot(ev.sigma_xy,tr['sigma_xy'])+0.10*min(ev.radius,tr['radius'])
                if dist<=gate:
                    eligible.append((dist,index))
            if eligible:
                _,idx=min(eligible)
                tr=tracks[idx]
                old_prec=tr['precision_xy'];new_prec=1/(ev.sigma_xy**2)
                total=old_prec+new_prec
                for k in ('x','y','radius'):
                    tr[k]=(tr[k]*old_prec+getattr(ev,k)*new_prec)/total
                tr['precision_xy']=total
                hp=tr['precision_height'];ep=1/(ev.sigma_height**2)
                tr['height']=(tr['height']*hp+ev.height*ep)/(hp+ep)
                tr['precision_height']=hp+ep
                tr['sigma_xy']=1/math.sqrt(total)
                tr['count']+=1
            else:
                tracks.append(dict(x=ev.x,y=ev.y,radius=ev.radius,height=ev.height,
                                   precision_xy=1/(ev.sigma_xy**2),
                                   precision_height=1/(ev.sigma_height**2),
                                   sigma_xy=ev.sigma_xy,count=1))
        self.obstacle_tracks=tracks

    def estimated_obstacle_height(self,cell:int,*,grid_n:int,cell_m:float):
        """Highest estimated cylinder over a ground cell, using observations only."""
        if not 0<=cell<grid_n*grid_n:raise IndexError(cell)
        cx=(cell%grid_n+.5)*cell_m
        cy=(cell//grid_n+.5)*cell_m
        answer=0.
        for tr in self.obstacle_tracks:
            dx=max(abs(cx-tr['x'])-cell_m/2.,0.)
            dy=max(abs(cy-tr['y'])-cell_m/2.,0.)
            if dx*dx+dy*dy<=tr['radius']**2:
                answer=max(answer,float(tr['height']))
        return answer

    def obstacle_voxel_map(self,*,grid_n:int,cell_m:float,z_levels):
        levels=np.asarray(z_levels,dtype=float)
        if grid_n<=0 or cell_m<=0 or levels.ndim!=1:raise ValueError("bad voxel geometry")
        centers=(np.arange(grid_n)+.5)*cell_m
        xx,yy=np.meshgrid(centers,centers,indexing='xy')
        voxel=np.zeros((len(levels),grid_n,grid_n),dtype=bool)
        for tr in self.obstacle_tracks:
            dx=np.maximum(np.abs(xx-tr['x'])-cell_m/2.,0.)
            dy=np.maximum(np.abs(yy-tr['y'])-cell_m/2.,0.)
            footprint=dx*dx+dy*dy<=tr['radius']**2
            for layer,z in enumerate(levels):
                if z<=tr['height']:voxel[layer]|=footprint
        return voxel

    def visible_summary(self,cell:int):
        """Actor/GCS observation from received evidence, never true occupancy."""
        if not 0<=cell<self.num_cells:raise IndexError(cell)
        cand=self.candidates.get(cell)
        return (tuple(float(x) for x in self.target_masses[cell]),
                tuple(float(x) for x in self.obstacle_masses[cell]),
                cand.status if cand else "unseen",cand.sprt_llr if cand else 0.)


@dataclass(frozen=True)
class EncodedBatch:
    events: tuple[Event,...]
    size_bytes: int


def _wire_bytes(events):
    # Conservative plain JSON overhead; headers reserved separately.
    return len(json.dumps([asdict(e) for e in events],separators=(",",":"),sort_keys=True).encode("utf-8"))+24


def packetize_events(events,*,payload_bytes=1024):
    if payload_bytes<64:raise ValueError("payload too small for evidence packet")
    groups=[];pending=[]
    for event in events:
        if not isinstance(event,Event):raise TypeError("observation event expected")
        new=pending+[event]
        if _wire_bytes(new)>payload_bytes:
            if not pending:
                raise ValueError("one observation event exceeds packet payload")
            groups.append(EncodedBatch(tuple(pending),_wire_bytes(pending)))
            pending=[event]
            if _wire_bytes(pending)>payload_bytes:
                raise ValueError("one observation event exceeds packet payload")
        else:pending=new
    if pending:groups.append(EncodedBatch(tuple(pending),_wire_bytes(pending)))
    return groups


class MultiLayerMapMetrics:
    """Evaluator-only class: actor/critic/GCS protocols never receive truth."""
    @staticmethod
    def evaluate(map_node:CooperativeWorldMap,target_occupancy,obstacle_occupancy,*,at_tick:int):
        target=np.asarray(target_occupancy,dtype=np.float64).ravel()
        obstacle=np.asarray(obstacle_occupancy,dtype=np.float64).ravel()
        if target.shape!=(map_node.num_cells,) or obstacle.shape!=(map_node.num_cells,):
            raise ValueError("truth shape must match ground-cell map")
        p_target=map_node.target_masses[:,0]+.5*map_node.target_masses[:,2]
        p_obstacle=map_node.obstacle_masses[:,0]+.5*map_node.obstacle_masses[:,2]
        # Threshold-group AP avoids an artificial ranking among untouched
        # cells whose initial pignistic probabilities are identical (0.5).
        def average_precision(y,p):
            positives=float(y.sum())
            if positives==0:return float('nan')
            order=np.argsort(-p,kind='stable')
            sorted_p=p[order];sorted_y=y[order]
            ends=np.r_[np.flatnonzero(sorted_p[:-1]!=sorted_p[1:]),len(y)-1]
            tp=np.cumsum(sorted_y)[ends]
            predicted=ends+1
            recall=tp/positives
            precision=tp/predicted
            return float(np.sum(np.diff(np.r_[0.,recall])*precision))
        y_target=(target>.5).astype(np.int32)
        bins=np.minimum((p_target*10).astype(np.int64),9)
        ece=0.
        for k in range(10):
            mask=bins==k
            if mask.any():
                ece+=(mask.mean()*abs(p_target[mask].mean()-y_target[mask].mean()))
        occupied_pred=map_node.obstacle_masses[:,0]>=map_node.ds_threshold
        occupied_truth=obstacle>.5
        obstacle_union=int(np.logical_or(occupied_pred,occupied_truth).sum())
        obstacle_iou=float(np.logical_and(occupied_pred,occupied_truth).sum()/obstacle_union) if obstacle_union else 1.
        detected=map_node.target_masses[:,0]>=map_node.ds_threshold
        tp=int((detected & (target>0.5)).sum()); fp=int((detected & (target<0.5)).sum())
        ages=np.maximum(0,at_tick-map_node.last_observed[map_node.last_observed>=0])
        return {"target_brier":float(np.square(p_target-target).mean()),
                "obstacle_brier":float(np.square(p_obstacle-obstacle).mean()),
                "coverage":float((map_node.coverage>0).mean()),
                "target_auprc":average_precision(y_target,p_target),
                "target_calibration_ece":float(ece),
                "obstacle_iou_2d":obstacle_iou,
                "target_precision":tp/max(tp+fp,1),
                "target_recall":tp/max(int((target>.5).sum()),1),
                "false_target_cells":fp,
                "mean_freshness_steps":float(ages.mean()) if len(ages) else float(at_tick)}
