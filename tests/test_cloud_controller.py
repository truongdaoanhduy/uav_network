"""Cloud continuation decisions must not depend on a live local Sandbox."""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_cloud_continuation import parent_spec, learner_result
from uav_network.cloud.credentials import Account, Credentials
from uav_network.cloud.kernel import kernel_metadata, kernel_script


class ControllerTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("uav_network.cloud.controller")
        except ModuleNotFoundError:
            self.fail("No cloud-side executor exists to wake a quota-paused lineage")

    def fixtures(self):
        m = self.module()
        parent = parent_spec()
        account = Account("selected", "private-launch-secret")
        credentials = Credentials([account], "private-wandb-secret", "owner", "project")
        spec = m.build_controller_spec(parent, "b" * 40, gpu_commit="c" * 40)
        return m, parent, account, credentials, spec

    def test_generated_controller_executes_cpu_branch_and_loads_private_launch_token(self):
        m, parent, account, credentials, spec = self.fixtures()
        metadata = kernel_metadata(spec)
        self.assertFalse(metadata["enable_gpu"])
        self.assertTrue(metadata["is_private"])
        self.assertEqual(set(metadata["dataset_sources"]),
                         {"owner/private-cpu-wandb", "owner/private-launch"})
        source = ast.parse(kernel_script(spec))
        env = {}
        # Execute actual bootstrap's CPU selection and runner-selection expressions.
        namespace = {"SPEC": spec, "os": SimpleNamespace(environ=env)}
        node = next(n for n in source.body if isinstance(n, ast.If))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<bootstrap>", "exec"), namespace)
        node = next(n for n in source.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "runner" for t in n.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<bootstrap>", "exec"), namespace)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(namespace["runner"], "uav_network.cloud.controller")
        self.assertTrue(spec["requires_kaggle_token"])
        self.assertNotIn(account.token, json.dumps(spec))
        self.assertNotIn(credentials.wandb_key, kernel_script(spec))

    def test_active_gpu_only_observes(self):
        m, parent, account, credentials, spec = self.fixtures()
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="running"), \
             patch.object(m, "read_controls", side_effect=AssertionError("active output read")), \
             patch.object(m, "submit_continuation", side_effect=AssertionError("duplicate launch")):
            state = m.controller_step(spec, deepcopy(spec["state"]), account, credentials, Path(d))
        self.assertEqual(state["phase"], "training")
        self.assertEqual(state["current_gpu_ref"], parent["kernel_ref"])

    def test_quota_pause_retries_later_and_pins_recovery_source(self):
        m, parent, account, credentials, spec = self.fixtures()
        calls = []
        def submit(p, result, output, **kw):
            calls.append(p)
            if len(calls) == 1:
                return {"status": "waiting_for_quota", "wait_reason": "insufficient_quota"}
            return {"status": "submitted", "kernel_ref": m.kernel_ref_for_segment(parent["kernel_ref"], 1)}
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": parent, "job-result.json": learner_result()}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m, "submit_continuation", side_effect=submit):
            state = m.controller_step(spec, deepcopy(spec["state"]), account, credentials, Path(d))
            self.assertEqual(state["phase"], "waiting_for_quota")
            state = m.controller_step(spec, state, account, credentials, Path(d))
        self.assertEqual(state["phase"], "training")
        self.assertTrue(state["current_gpu_ref"].endswith("-c01"))
        self.assertEqual(calls[-1]["commit"], "c" * 40)
        self.assertEqual(calls[-1]["timeout_seconds"], 39600)

    def test_existing_failed_child_is_adopted_and_never_overwritten(self):
        m, parent, account, credentials, spec = self.fixtures()
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": parent, "job-result.json": learner_result()}), \
             patch.object(m, "_owned_kernel_exists", return_value=True), \
             patch.object(m, "submit_continuation", side_effect=AssertionError("overwrite")):
            state = m.controller_step(spec, deepcopy(spec["state"]), account, credentials, Path(d))
        self.assertTrue(state["current_gpu_ref"].endswith("-c01"))
        self.assertEqual(state["phase"], "training")

    def test_final_cpu_is_not_complete_until_both_publications_succeed(self):
        m, parent, account, credentials, spec = self.fixtures()
        cpu = {"stage": "visualize", "account": account.name, "kernel_ref": "owner/final-cpu",
               "root_kernel_ref": parent["root_kernel_ref"], "artifacts": {"masac": "owner/project/policy:v5"},
               "cpu_eval": parent["cpu_eval"], "runs": {"masac": {"id": "cpu", "name": "cpu"}}}
        state = {**spec["state"], "current_cpu_ref": cpu["kernel_ref"], "phase": "visualizing"}
        result = {"masac": {"episodes": 2, "device": "cpu",
                           "checkpoint_artifact": cpu["artifacts"]["masac"],
                           "records": [{"episode": 1, "seed": 10044}, {"episode": 2, "seed": 10045}],
                           "publication": {"wandb": {"status": "ok"},
                                           "huggingface": {"status": "ok", "revision": "f"*40,
                                               "files": {k: {"url": "https://huggingface.co/owner/model/resolve/"+ "f"*40 +"/"+k}
                                                         for k in ("video", "image", "metrics")}}}}}
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": cpu, "visualization-result.json": result}):
            done = m.controller_step(spec, deepcopy(state), account, credentials, Path(d))
        self.assertEqual(done["phase"], "complete")
        result["masac"]["publication"]["wandb"]["status"] = "failed"
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": cpu, "visualization-result.json": result}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m, "submit_cpu_attempt", return_value={"kernel_ref": "owner/cpu-retry"}):
            pending = m.controller_step(spec, deepcopy(state), account, credentials, Path(d))
        self.assertNotEqual(pending["phase"], "complete")
        self.assertEqual(pending["current_cpu_ref"], "owner/cpu-retry")

    def test_invalid_terminal_summary_blocks_without_submission(self):
        m, parent, account, credentials, spec = self.fixtures()
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="error"), \
             patch.object(m, "read_controls", return_value={"job.json": parent, "job-result.json": {}}), \
             patch.object(m, "submit_continuation", side_effect=AssertionError("invalid resume")):
            state = m.controller_step(spec, deepcopy(spec["state"]), account, credentials, Path(d))
        self.assertEqual(state["phase"], "blocked")

    def test_controller_generation_carries_actual_progress_and_adopts_existing_successor(self):
        m, parent, account, credentials, spec = self.fixtures()
        state = {**spec["state"], "phase": "waiting_for_quota", "current_gpu_ref": "owner/progress-c07"}
        next_spec = m.build_controller_spec(parent, spec["commit"], gpu_commit=spec["gpu_commit"],
                                           state=state, generation=1)
        self.assertNotEqual(next_spec["kernel_ref"], spec["kernel_ref"])
        self.assertLessEqual(len(next_spec["kernel_ref"]), 50)
        self.assertEqual(next_spec["state"], state)
        with TemporaryDirectory() as d, patch.object(m, "_owned_kernel_exists", return_value=True), \
             patch.object(m, "kernel_status", return_value="running"), \
             patch.object(m, "run_kaggle", side_effect=AssertionError("overwrite controller")):
            adopted = m.submit_controller(next_spec, Path(d), account, credentials)
        self.assertEqual(adopted["status"], "adopted")
        self.assertEqual(adopted["kernel_ref"], next_spec["kernel_ref"])

    def test_controller_ref_is_unique_across_long_truncated_roots(self):
        m = self.module()
        self.assertNotEqual(m.controller_ref("owner/"+"a"*70+"1", 0),
                            m.controller_ref("owner/"+"a"*70+"2", 0))

    def test_cpu_progress_can_span_more_than_three_sessions(self):
        m, parent, account, credentials, spec = self.fixtures()
        cpu = {"stage": "visualize", "account": account.name, "kernel_ref": "owner/final-cpu",
               "root_kernel_ref": parent["root_kernel_ref"], "artifacts": {"masac": "owner/project/policy:v5"},
               "cpu_eval": parent["cpu_eval"], "runs": {"masac": {"id": "cpu", "name": "cpu"}}}
        progress = {"artifacts": cpu["artifacts"], "cpu_eval": cpu["cpu_eval"],
                    "records": {"masac": [{"episode": 1, "seed": 10044}]}}
        state = {**spec["state"], "current_cpu_ref": cpu["kernel_ref"], "phase": "visualizing",
                 "cpu_attempt": 4, "evaluation_records_completed": 0}
        submitted = []
        def submit(cpu, *args):
            submitted.append(cpu)
            return {"kernel_ref": cpu["kernel_ref"]}
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": cpu, "evaluation-progress.json": progress}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m, "submit_cpu_attempt", side_effect=submit):
            result = m.controller_step(spec, state, account, credentials, Path(d))
        self.assertEqual(result["phase"], "visualizing")
        self.assertEqual(submitted[0]["completed_records"], progress["records"])
        self.assertEqual(result["cpu_attempt"], 5)

    def test_ambiguous_gpu_submission_is_reconciled_without_a_second_push(self):
        m, parent, account, credentials, spec = self.fixtures()
        target = m.kernel_ref_for_segment(parent["kernel_ref"], 1)
        state = {**spec["state"], "submission_intent": {"ref": target, "kind": "gpu"}}
        with TemporaryDirectory() as d, patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m, "submit_continuation", side_effect=AssertionError("repeated uncertain push")):
            waiting = m.controller_step(spec, state, account, credentials, Path(d))
        self.assertEqual(waiting["phase"], "submission_uncertain")
        with TemporaryDirectory() as d, patch.object(m, "_owned_kernel_exists", return_value=True), \
             patch.object(m, "kernel_status", return_value="running"):
            adopted = m.controller_step(spec, waiting, account, credentials, Path(d))
        self.assertEqual(adopted["current_gpu_ref"], target)
        self.assertNotIn("submission_intent", adopted)

    def test_real_session_loop_hands_state_to_next_cpu_generation(self):
        m, parent, account, credentials, spec = self.fixtures()
        submitted = []
        run = SimpleNamespace(summary={}, log=lambda *a: None, finish=lambda **kw: None)
        sdk = SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
        import os, sys
        def submit(successor, *args):
            submitted.append(successor)
            return {"kernel_ref": successor["kernel_ref"]}
        with TemporaryDirectory() as d, patch.dict(sys.modules, {"wandb": sdk}), \
             patch.dict(os.environ, {"WANDB_API_KEY": credentials.wandb_key,
                                     "KAGGLE_API_TOKEN": account.token, "UAV_JOB_STARTED_AT": "0"}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m.time, "monotonic", return_value=39001), \
             patch.object(m, "submit_controller", side_effect=submit):
            result = m.run_controller(spec, Path(d))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["generation"], 1)
        self.assertEqual(submitted[0]["state"]["current_gpu_ref"], parent["kernel_ref"])
        self.assertEqual(result["controller_status"], "handed_off")

    def test_secret_in_nested_parent_is_rejected_before_submission(self):
        m, parent, account, credentials, spec = self.fixtures()
        spec["parent"]["accidental_secret"] = account.token
        with TemporaryDirectory() as d, patch.object(m, "run_kaggle", side_effect=AssertionError("secret push")):
            with self.assertRaisesRegex(ValueError, "credentials"):
                m.submit_controller(spec, Path(d), account, credentials)

    def test_controller_session_retries_a_transient_handoff_failure(self):
        m, parent, account, credentials, spec = self.fixtures()
        submitted = []
        run = SimpleNamespace(summary={}, log=lambda *a: None, finish=lambda **kw: None)
        sdk = SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
        import os, sys
        def submit(successor, *args):
            submitted.append(successor)
            if len(submitted) == 1:
                raise TimeoutError("temporary API failure")
            return {"kernel_ref": successor["kernel_ref"]}
        with TemporaryDirectory() as d, patch.dict(sys.modules, {"wandb": sdk}), \
             patch.dict(os.environ, {"WANDB_API_KEY": credentials.wandb_key,
                                     "KAGGLE_API_TOKEN": account.token, "UAV_JOB_STARTED_AT": "0"}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m.time, "monotonic", return_value=39001), patch.object(m.time, "sleep"), \
             patch.object(m, "submit_controller", side_effect=submit):
            result = m.run_controller(spec, Path(d))
        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0]["kernel_ref"], submitted[1]["kernel_ref"])
        self.assertEqual(result["controller_status"], "handed_off")

    def test_downloaded_gpu_cannot_change_the_original_scientific_config(self):
        m, parent, account, credentials, spec = self.fixtures()
        changed = {**parent, "overrides": ["runtime.episodes=10000", "world.num_targets=1"]}
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": changed, "job-result.json": learner_result()}), \
             patch.object(m, "_owned_kernel_exists", return_value=True), \
             patch.object(m, "submit_continuation", side_effect=AssertionError("configuration drift")):
            result = m.controller_step(spec, deepcopy(spec["state"]), account, credentials, Path(d))
        self.assertEqual(result["phase"], "blocked")

    def test_pre_push_credential_failure_retries_instead_of_waiting_for_nonexistent_child(self):
        m, parent, account, credentials, spec = self.fixtures()
        from uav_network.cloud import continuation
        with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
             patch.object(m, "read_controls", return_value={"job.json": parent, "job-result.json": learner_result()}), \
             patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(continuation, "account_quotas", return_value=[{"name": account.name, "ok": True, "remaining_hours": 20}]), \
             patch.object(continuation, "verify_credential_dataset", side_effect=TimeoutError("credentials offline")), \
             patch.object(continuation, "run_kaggle", side_effect=AssertionError("must not push")):
            state = deepcopy(spec["state"])
            with self.assertRaises(TimeoutError):
                m.controller_step(spec, state, account, credentials, Path(d))
            self.assertNotIn("submission_intent", state)
            with self.assertRaises(TimeoutError):
                m.controller_step(spec, state, account, credentials, Path(d))

    def test_existing_failed_controller_is_preserved_and_not_accepted_as_handoff(self):
        m, parent, account, credentials, spec = self.fixtures()
        with TemporaryDirectory() as d, patch.object(m, "_owned_kernel_exists", return_value=True), \
             patch.object(m, "kernel_status", return_value="error"), \
             patch.object(m, "run_kaggle", side_effect=AssertionError("overwrite failed successor")):
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                m.submit_controller(spec, Path(d), account, credentials)

    def test_malformed_control_shapes_are_blocked_instead_of_endless_service_retries(self):
        m, parent, account, credentials, spec = self.fixtures()
        bad_controls = [{"job.json": []}, {"job.json": parent, "job-result.json": []}]
        cpu = {"stage": "visualize", "account": account.name, "kernel_ref": "owner/cpu",
               "root_kernel_ref": parent["root_kernel_ref"], "cpu_eval": parent["cpu_eval"],
               "artifacts": {"masac": "owner/project/model:v1"}}
        bad_controls += [{"job.json": cpu, "visualization-result.json": {"masac": []}},
                         {"job.json": cpu, "evaluation-progress.json": []},
                         {"job.json": cpu, "visualization-result.json": {"masac": {"publication": []}}},
                         {"job.json": cpu, "visualization-result.json": {"masac": {"publication": {
                             "huggingface": {"files": {"video": []}}}}}}]
        for controls in bad_controls:
            with self.subTest(controls=controls):
                state = deepcopy(spec["state"])
                if controls.get("job.json") == cpu:
                    state["current_cpu_ref"] = "owner/cpu"
                with TemporaryDirectory() as d, patch.object(m, "kernel_status", return_value="complete"), \
                     patch.object(m, "read_controls", return_value=controls):
                    result = m.controller_step(spec, state, account, credentials, Path(d))
                self.assertEqual(result["phase"], "blocked")

    def test_uncertain_controller_push_is_not_repeated_when_listing_temporarily_lags(self):
        m, parent, account, credentials, spec = self.fixtures()
        import os, sys
        run = SimpleNamespace(summary={}, log=lambda *a: None, finish=lambda **kw: None)
        sdk = SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw)
        pushes = []
        clocks = iter([39001, 39002, 39003, 39004, 39005, 39500, 39501, 39502, 39503])
        def push(successor, output, *args):
            pushes.append(successor["kernel_ref"])
            (Path(output)/"controller-submission.json").write_text(json.dumps(
                {"ref": successor["kernel_ref"], "phase": "pushing"}))
            raise TimeoutError("response lost")
        with TemporaryDirectory() as d, patch.dict(sys.modules, {"wandb": sdk}), \
             patch.dict(os.environ, {"WANDB_API_KEY": credentials.wandb_key,
                                     "KAGGLE_API_TOKEN": account.token, "UAV_JOB_STARTED_AT": "0"}), \
             patch.object(m.time, "monotonic", side_effect=lambda: next(clocks, 39599)), \
             patch.object(m.time, "sleep"), patch.object(m, "_owned_kernel_exists", return_value=False), \
             patch.object(m, "submit_controller", side_effect=push):
            with self.assertRaises(RuntimeError):
                m.run_controller(spec, Path(d))
        self.assertEqual(len(pushes), 1, "An ambiguous controller push must be reconciled, never repeated")
