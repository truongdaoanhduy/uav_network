# Packet channel model and scenario contract

This project uses the analytic `a2a` channel from UavNetSim revision
`04daafb815eb377409b40b285574eeb62b9a8d58`, with a Warp implementation of the same
mission packet model. The version pin is in
[`communication/upstream.py`](../uav_network/communication/upstream.py). The
selected backend does not invoke Sionna ray tracing. Features advertised by the
latest upstream README, or by other upstream channel modes, are not evidence
that this mission enables them.

The active profiles
[`comparison-20k.yaml`](../configs/comparison-20k.yaml),
[`motion-gpu.yaml`](../configs/motion-gpu.yaml), and
[`motion-pql.yaml`](../configs/motion-pql.yaml) now describe **6 UAVs, 35 targets,
25 obstacles, and a 1000 m map**. Their previous scenario had 25 targets and
15 obstacles. Scenario identity and output/tracking names distinguish these
experiments. Saved checkpoints retain their saved configuration; prior results
must not be relabelled as measurements of the enlarged scenario. Target/obstacle
counts can change policy observation/state dimensions.

## Implemented effects

| Effect | Native / Warp status | Contract and source |
|---|---|---|
| Geometric LOS/NLOS | Present in both | Radio rays intersect obstacle polygons. Native uses building volumes; Warp clips a segment against convex polygons. See [upstream airspace][airspace] and [`gpu/phy.py:los`](../uav_network/communication/gpu/phy.py). |
| Obstacle height | Present under this mission's fixed-height geometry | Scene buildings extend from ground to UAV altitude + 20 m. UAVs have common fixed altitude; GCS has height 0. Thus every polygon intersection blocks these mission links. Warp's 2D LOS test is equivalent for this geometry, not a general variable-height building model. |
| Three-dimensional link distance | Present | UAV/UAV links have equal height; UAV/GCS distance includes altitude squared. Native uses `math.dist`; Warp adds the height difference before taking the square root. |
| Distance/frequency attenuation | Present | Free-space LOS and urban NLOS formulas below match [pinned A2A][a2a]. |
| Additional obstruction loss | Present | `obstacle_extra_loss_db` applies once to an NLOS link, in addition to the urban model's 12 dB. It is not multiplied by the number of intersected buildings. Native [`energy.py`](../uav_network/communication/energy.py) / Warp `p.nlos_scale`. |
| Thermal noise and receiver noise figure | Present | Bandwidth-integrated noise in watts; see [pinned config][config] and [`gpu/state.py`](../uav_network/environment/gpu/state.py). |
| Concurrent packet interference | Present | Overlapping airtime is weighted by its fraction of the desired frame, including ACK transmissions. SINR determines reception. See [pinned channel][channel] and Warp `evaluate`. |
| Hidden senders and CCA | Present | Each sender senses aggregate received power from other active transmitters against `cca_threshold_dbm`. CCA is not an ideal common busy flag. |
| Self transmission during reception | Present as an interference approximation | A receiver's own concurrent TX uses gain 1 rather than a remote path loss. This strongly suppresses decoding; it is not a separate hard half-duplex state machine. Both implementations follow the same convention. |
| Channel separation | One shared mission channel | Native supports overlapping channel IDs; mission DATA and ACK packets all use channel 1. Warp therefore treats all frames as co-channel. There is no learned channel-selection action. |
| CSMA/CA, interframe spaces, contention | Present | Slot 20 us, SIFS 10 us, DIFS 50 us; seeded binary exponential backoff and interrupted-backoff handling mirror [pinned CSMA/CA][csma]. |
| ACKs, timeout, retries | Present | Successful DATA reception triggers ACK after SIFS. Missing ACK triggers retry up to the configured limit. ACK has its own power, airtime, interference, and battery check. |
| Packet airtime, headers and queues | Present | DATA includes 464 header bits; ACK is 240 bits. Packet lifetime, report reassembly, original report TTL, receiver admission and finite custody buffers are enforced. |
| Variable TX power and radiation energy | Present | The actor's normalized third control selects 0.1–0.4 W on packet injection. Retries retain that packet's selected power; every emitted DATA/ACK consumes airtime times power, including failed attempts. |
| Source/receiver motion | Discrete mission positions with gain caching | XY positions update once per mission step. Native and Warp refresh a pair's channel gain after 100 ms or movement of at least 1 m. Baseline reception evaluates against the then-current cached geometry, rather than integrating a continuous trajectory over the frame. |

The 500 m peer/GCS admission radii are an additional policy-routing constraint in
XY coordinates. They do not replace the propagation loss or guarantee successful
reception, and increasing TX power does not expand this admission mask.

## Exact analytic formulas

Let `d = max(1 m, ||x_tx - x_rx||_3D)`, carrier frequency be `f`, and
`c = 299792458 m/s`. With a 1 m reference distance:

\[
G_0=\left(\frac{c}{4\pi f\cdot1\,\mathrm m}\right)^2,
\qquad
G_{LOS}=G_0\left(\frac d{1\,\mathrm m}\right)^{-2},
\]

For `nlos_model=urban`, the exponent/excess loss are `(3.0, 12 dB)`; for
`nlos_model=suburban`, they are `(2.6, 8 dB)`. Thus

\[
G_{NLOS}=G_0\left(\frac d{1\,\mathrm m}\right)^{-\eta}
10^{-(L_{NLOS}+L_{extra})/10}.
\]

These are deterministic received-power gains, not a validated 3GPP urban channel.
Pinned [A2A model constants and formulas][a2a] are the authority for the selected
analytic model; Warp [`gain`](../uav_network/communication/gpu/phy.py) implements
the same equations.

For noise density `N0` in dBm/Hz, bandwidth `B` in Hz, and noise figure `NF` in dB:

\[
N=10^{(N_0+10\log_{10}B+NF-30)/10}\quad\mathrm W.
\]

The current values `N0=-174`, `B=22e6`, `NF=7` give
`N = 4.389577092931546e-13 W`. For desired frame `i` and interfering frame `j`,
let `a_ij` be the overlap duration divided by frame `i`'s duration. Then:

\[
SINR_i=\frac{P_i G_{i,r}}
 {N+\sum_{j\ne i}a_{ij}P_jG_{j,r}},
\qquad 10\log_{10}(SINR_i)\ge6\,\mathrm{dB}
\]

is the default success criterion. The sum includes all overlapping mission
frames; when transmitter `j` is receiver `r`, `G_jr=1`. Overlap weighting is a
packet-level approximation, not symbol-by-symbol decoding or a bit-error model.

## Relation to the paper and deliberately absent effects

The [UavNetSim-v1 paper, sections III-C and III-F][paper] describes ACK/retry and
radiation energy `E_comm = P_tx * t_packet` (equation 3). Section IV lists
receiver/circuit/signal-processing energy as possible extensions. Their absence
here is therefore not a missing implementation of equation 3. This project also
uses its own documented speed/acceleration propulsion surrogate; it does not
claim to reproduce the paper's constant-speed propulsion equation.

The paper omits coding/modulation detail. Its Table I uses fixed noise
`4e-11 W`, about **91.1 times / 19.60 dB higher** than the current pinned thermal
noise calculation. These profiles therefore do not reproduce that table's
experimental noise setting. No automatic noise change is justified by the
comparison; it would change the experiment.

The selected analytic backend has no random shadowing, fast fading, Doppler,
frequency-selective multipath, reflection/diffraction ray tracing, or radio
receiver/circuit power model. No uncalibrated coefficients were added for these
effects. Pinned upstream has other [Sionna channel modes][sionna], but selecting
`a2a` explicitly bypasses them.

Commit `5419916` omitted finite-speed flight time: native
[`Channel._deliver`][channel] waited for airtime only and Warp scheduled PHY
reception at frame end. The current profiles enable `propagation_delay_enabled`
and use `t_prop=d/c`, with `c=299792458 m/s`. Each emitted frame freezes its
source-to-observer propagation delays. Native shifts the frame interval to the
observer before CCA/SINR overlap evaluation and delivers at the receiver-relative
arrival end; Warp stores the same per-frame/per-observer delays and uses them for
CCA, interference overlap, DATA delivery and ACK delivery. ACK timeout includes a
conservative round-trip allowance based on the maximum map link so a valid
long-link ACK cannot cause a premature retry. The generic configuration default
remains disabled so saved legacy configurations preserve their prior timing.
At 100 m, `d/c` is about 0.333564 us. This is packet-level propagation timing,
not continuous-wave mobility integration: source/receiver geometry is frozen at
emission, consistent with the existing discrete mission-position model.

## Verified defects and numerical boundaries

- **Unsupported NLOS configuration:** baseline validation accepted
  `nlos_model=free_space`, but [pinned NLOS models][a2a] contain `urban` and
  `suburban`, not `free_space`. A blocked native link therefore raised
  `Unsupported NLoS A2A model`. This project supports the pinned upstream
  `urban` and `suburban` options in both native and Warp backends, and rejects
  unsupported values before simulator startup.
- **Empty-medium CCA boundary:** at a configured threshold of -200 dBm, native
  returns idle for zero active frames, while baseline Warp compared a finite
  -200 dBm sentinel to the threshold and returned busy. The minimal correction
  is an unconditional idle result for zero sensed power; ordinary -82 dBm
  behavior is unchanged.
- **Exact obstacle tangencies:** an audit found 3 native-blocked/Warp-clear
  classifications among 150 constructed tangent/near-tangent rays, and none
  among 2500 random rays. These were exact-boundary floating-point cases. The
  current polygon algorithms should not be described as bitwise identical on
  boundary contact. A shared conservative contact convention and explicit
  tangency regressions are preferable to an arbitrary large obstruction margin.

## Verification evidence and scale

The read-only probe in `outputs/channel-audit-20261016/channel-probe.json`
compared 378 directed off-diagonal link gains across 0.9/2.4/5.8 GHz and
0/7/25 dB extra obstruction loss. Maximum relative error was
`5.61e-16`. It also checked the LOS corpus above and reset the enlarged 35/25
scene at seeds `0, 1, 44, 99, 2000000044, 2000000073`: native/Warp UAV, target
and obstacle coordinates were exactly equal and all Warp error codes were zero.

At that baseline configuration, one CPU Warp world used **475500 bytes**, with
1470 report sessions and 396 packet slots. The disabled optional report-trace
payload had shape `(1, 0, 9)` and consumed zero payload bytes. Session/target
arrays scale with the target count; packet capacity follows finite radio queues.
This probe does not establish GPU training throughput or guarantee placement
success for every seed at arbitrarily denser obstacle settings. Placement and
capacity failures are explicit errors.

Regressions in [`test_gpu_packets.py`](../tests/test_gpu_packets.py),
[`test_gpu_physics.py`](../tests/test_gpu_physics.py),
[`test_communication.py`](../tests/test_communication.py),
[`test_channel_models.py`](../tests/test_channel_models.py), and
[`test_propagation.py`](../tests/test_propagation.py) cover native/Warp
contention, hidden senders, obstacle NLOS, DATA/ACK timing, retries, custody,
battery denial, finite queues, propagation arrival timing, receiver-relative
CCA/interference, and enlarged-scene reset/step parity. CUDA-specific tests are
separate from Warp-on-CPU tests and must not be reported as passed on a host
without a CUDA device.

## Primary sources

The checked-in `third_party/UavNetSim` files at the pinned revision are the
source of truth for compatibility. Links below identify that exact revision,
not the moving default branch.

[paper]: https://arxiv.org/html/2507.09852v1
[a2a]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/phy/a2a.py
[channel]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/phy/channel.py
[config]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/utils/config.py
[sionna]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/phy/sionna_rt.py
[airspace]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/scene/airspace.py
[csma]: https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/04daafb815eb377409b40b285574eeb62b9a8d58/mac/csma_ca.py
