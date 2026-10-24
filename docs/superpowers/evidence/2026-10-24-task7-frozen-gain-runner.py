import json, os, subprocess, sys, tarfile, time, statistics
from pathlib import Path
ROOT = Path("/tmp/uav-cache-ab")
BASE = "6062744b6626fcc73f55fab8b7f321da74a5f1d9"
CANDIDATE = "5fcb272364e7b574280cc4aaa9aea1c8820be088"
subprocess.run(["git","clone","--quiet","https://github.com/truongdaoanhduy/uav_network.git",str(ROOT)],check=True)
subprocess.run(["git","checkout","--quiet",CANDIDATE],cwd=ROOT,check=True)
subprocess.run(["bash","setup.sh","--system-site-packages"],cwd=ROOT,
               env=dict(os.environ,PYTHON=sys.executable),check=True)
python = str(ROOT/".venv/bin/python")
subprocess.run([python,"-m","pip","install","-q","warp-lang==1.18.0","pytest","pytest-subtests"],check=True)
bench = Path("/kaggle/working/frozen_gain_bench.py")
bench.write_text("import hashlib, json, os, sys, time\nfrom pathlib import Path\nimport torch\nimport warp as wp\nfrom uav_network.configuration import load_config\nfrom uav_network.environment.gpu.vector import WarpVectorEnv\n\ntorch.set_num_threads(1)\nlabel, repeat, output = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])\nbase = load_config(\"configs/comparison-20k-hybrid75-1m.yaml\")\nbase.tracking.enabled = False\nbase.runtime.num_envs = 4096\nbase.runtime.device = \"cuda\"\nbase.runtime.seed = 44\ndef hashes(env):\n    return {name: hashlib.sha256(wp.to_torch(value).cpu().contiguous().numpy().tobytes()).hexdigest()\n            for name,value in sorted(vars(env.s).items()) if isinstance(value,wp.array)}\n\ndef force_reports(env, distinct):\n    s, p = env.s, env.p\n    for source in range(p.n):\n        target = source if distinct else 0\n        wp.to_torch(s.report_present)[:,source,target] = 1\n        wp.to_torch(s.known)[:,source,target] = 1\n        wp.to_torch(s.report_created)[:,source,target] = 0\n        wp.to_torch(s.report_order)[:,source,target] = source+1\n        wp.to_torch(s.report_origin)[:,source,target] = source\n        wp.to_torch(s.report_hops)[:,source,target] = 0\n\ndef run_case(name, size, peer, steps):\n    from copy import deepcopy\n    c = deepcopy(base)\n    c.network.report_bytes = size\n    c.validate()\n    env = WarpVectorEnv(c, device=\"cuda:0\")\n    try:\n        env.reset(seed=44)\n        force_reports(env, distinct=peer)\n        action = {\"continuous\": torch.zeros((4096,c.world.num_uavs,3), device=\"cuda:0\"),\n                  \"destination\": torch.full((4096,c.world.num_uavs),\n                                            1 if peer else c.world.num_uavs,\n                                            device=\"cuda:0\", dtype=torch.int64)}\n        action[\"continuous\"][:,:,0] = .35\n        action[\"continuous\"][:,:,1] = -.2\n        action[\"continuous\"][:,:,2] = 1\n        initial = {name: wp.to_torch(value).clone() for name,value in vars(env.s).items()\n                   if isinstance(value,wp.array)} if size == 1024 else None\n        env.step(action)\n        torch.cuda.synchronize(0)\n        elapsed = 0.0\n        for _ in range(steps):\n            if initial is not None:\n                for array_name,value in initial.items():\n                    wp.to_torch(getattr(env.s,array_name)).copy_(value)\n                torch.cuda.synchronize(0)\n            start = time.perf_counter()\n            env.step(action)\n            torch.cuda.synchronize(0)\n            elapsed += time.perf_counter()-start\n        row = {\"label\": label, \"repeat\": repeat, \"case\": name,\n               \"num_envs\":4096, \"seed\":44, \"report_bytes\":size,\n               \"steps\":steps, \"warmup_steps\":1, \"seconds\":elapsed,\n               \"transitions_per_s\":4096*steps/elapsed,\n               \"generated_packets\":int(wp.to_torch(env.s.packet_counts)[:,0].sum().item()),\n               \"errors\":int(wp.to_torch(env.s.errors).count_nonzero().item()),\n               \"gpu_name\":torch.cuda.get_device_name(0),\n               \"hashes\":hashes(env)}\n        assert row[\"errors\"] == 0\n        assert row[\"generated_packets\"] > 0\n        print(json.dumps({k:v for k,v in row.items() if k!=\"hashes\"}), flush=True)\n        return row\n    finally:\n        env.close()\n\nrows = [run_case(\"active_1mb_peer\", 1000000, True, 1)]\noutput.write_text(json.dumps(rows,indent=2))\n")
trees = {}
for label, commit in [("baseline",BASE),("candidate",CANDIDATE)]:
    folder = Path("/tmp/uav-"+label)
    folder.mkdir()
    archive = subprocess.check_output(["git","archive",commit],cwd=ROOT)
    import io
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data")
    (folder/"third_party").symlink_to(ROOT/"third_party",target_is_directory=True)
    trees[label] = folder
all_rows=[]
order=["baseline","candidate","candidate","baseline","baseline","candidate"]
for index,label in enumerate(order):
    repeat = sum(1 for x in order[:index] if x==label)
    result = Path(f"/kaggle/working/cache-{index}-{label}.json")
    print("START",label,repeat,flush=True)
    subprocess.run([python,str(bench),label,str(repeat),str(result)],cwd=trees[label],
                   env=dict(os.environ,PYTHONPATH=str(trees[label])),check=True)
    all_rows.extend(json.loads(result.read_text()))
    Path("/kaggle/working/frozen-gain-ab-partial.json").write_text(json.dumps(all_rows,indent=2))
comparisons={}
for case in {r["case"] for r in all_rows}:
    a=[r for r in all_rows if r["case"]==case and r["label"]=="baseline"]
    b=[r for r in all_rows if r["case"]==case and r["label"]=="candidate"]
    mismatches=[key for key in a[0]["hashes"] if any(r["hashes"][key]!=a[0]["hashes"][key] for r in a+b)]
    av=statistics.median(r["transitions_per_s"] for r in a)
    bv=statistics.median(r["transitions_per_s"] for r in b)
    comparisons[case]={"baseline_median":av,"candidate_median":bv,
                       "speedup_percent":100*(bv/av-1),"mismatched_arrays":mismatches,
                       "baseline_rates":[r["transitions_per_s"] for r in a],
                       "candidate_rates":[r["transitions_per_s"] for r in b]}
result={"baseline_commit":BASE,"candidate_commit":CANDIDATE,
        "order":order,"cuda_device_count":subprocess.check_output(
        [python,"-c","import torch;print(torch.cuda.device_count())"],text=True).strip(),
        "cases":comparisons,"rows":all_rows}
Path("/kaggle/working/frozen-gain-ab.json").write_text(json.dumps(result,indent=2))
print("RESULT",json.dumps(comparisons),flush=True)
assert all(not r["mismatched_arrays"] for r in comparisons.values()), "Simulator state mismatch"

# Acceptance is evidence-based; a negative performance result remains a successful diagnostic.
result["performance_gate"] = {name: {"minimum_gain_percent": 2.0,
    "accepted": row["speedup_percent"] >= 2.0 and min(row["candidate_rates"]) > max(row["baseline_rates"])}
    for name,row in comparisons.items()}
result["fixture"] = {"num_envs":4096,"num_uavs":6,"num_targets":35,"num_obstacles":25,
    "report_bytes":1000000,"buffer_bytes":3000000,"dt_s":1.0,"max_steps":1800,"seed":44,
    "warmup_steps":1,"timed_steps":1,"traffic":"active_peer","array_hash_scope":"all_initialized_Warp_arrays"}
versions = subprocess.check_output([python,"-c","import json,torch,warp;print(json.dumps(dict(torch=torch.__version__,warp=warp.__version__,cuda=torch.version.cuda)))"],text=True)
result["versions"] = json.loads(versions)
test_code = """import importlib.util,sys,unittest
sys.path.insert(0,"tests")
import test_gpu_packets as packets
import test_propagation as propagation
packets.GPUPacketTests.device="cuda:0"
propagation.PropagationTests.device="cuda:0"
suite=unittest.defaultTestLoader.loadTestsFromTestCase(packets.GPUPacketTests)
suite.addTest(propagation.PropagationTests("test_frozen_step_cca_matches_generic_state_across_arrivals_and_movement"))
r=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not r.wasSuccessful())
"""
oracle=subprocess.run([python,"-c",test_code],cwd=trees["candidate"],
    env=dict(os.environ,PYTHONPATH=str(trees["candidate"])),capture_output=True,text=True)
Path("/kaggle/working/frozen-gain-cuda-oracle.log").write_text(oracle.stdout+oracle.stderr)
result["cuda_oracle_exit_code"]=oracle.returncode
result["cuda_oracle_tail"]=(oracle.stdout+oracle.stderr)[-1400:]
Path("/kaggle/working/frozen-gain-ab.json").write_text(json.dumps(result,indent=2))
print("PERFORMANCE_GATE",json.dumps(result["performance_gate"]),flush=True)
print("CUDA_ORACLE",oracle.returncode,result["cuda_oracle_tail"],flush=True)
raise SystemExit(oracle.returncode)
