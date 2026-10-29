"""Dashboard targets persist without depending on the training service process."""
from contextlib import ExitStack, chdir
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.experiments.dashboard import PANELS, create_dashboard


@unittest.skipUnless(importlib.util.find_spec("wandb_workspaces"), "cloud SDK is optional")
class DashboardTests(unittest.TestCase):
    def check_personal_workspace(self, exists):
        username = "researcher"
        personal_name = f"nw-nwuser{username}-w"
        saved = {"id": "saved-id", "name": "nw-existing-v", "displayName": "Saved layout",
                 "spec": {"unchanged": True}}
        stored = {"nw-existing-v": dict(saved)}
        if exists:
            stored[personal_name] = {"id": "personal-id", "name": personal_name,
                                     "displayName": "Researcher's workspace"}
        writes = []

        def query(document, variables=None):
            if "viewer" in document:
                return {"viewer": {"username": username}}
            if "allViews" in document:
                node = stored.get(variables["viewName"])
                return {"project": {"allViews": {"edges": [{"node": node}] if node else []}}}
            if "runs(" in document:
                node = {"name": "older-run", "config": json.dumps({
                    "tracking": {"value": {"group": "uav-network-2d", "enabled": True}}}),
                    "summaryMetrics": json.dumps({"episode/return": 2})}
                return {"project": {"runs": {"edges": [{"node": node}],
                                            "pageInfo": {"hasNextPage": False, "endCursor": None}}}}
            if "upsertView" in document:
                payload = variables["input"]
                writes.append(dict(payload))
                node = stored.setdefault(payload["name"], {"id": f"view-{len(stored)}"})
                node.update(name=payload["name"], displayName=payload["displayName"],
                            spec=json.loads(payload["spec"]))
                return {"upsertView": {"view": {"id": node["id"], "name": node["name"]}}}
            raise AssertionError("Unexpected workspace API operation")

        with TemporaryDirectory() as directory, chdir(directory), ExitStack() as stack:
            config = Config()
            config.tracking.credentials_file = str(Path(directory) / "absent-key.txt")
            stack.enter_context(patch.dict(os.environ, {
                "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}))
            stack.enter_context(patch("wandb.Api", side_effect=RuntimeError("training service is busy")))
            # The external HTTP boundary is replaced; all workspace serialization is real.
            stack.enter_context(patch("uav_network.experiments.dashboard._graphql", side_effect=query, create=True))
            for _ in range(2):
                result = create_dashboard(config)
                self.assertEqual(result["workspace_url"], "https://wandb.ai/owner/project/workspace")
                self.assertEqual(result["workspace_type"], "personal")
                self.assertEqual(result["pinned_runs"], [])
                self.assertEqual(result["panel_count"], sum(len(items) for items in PANELS.values()) + 1)
                self.assertEqual(json.loads(Path("outputs/wandb/workspace.json").read_text()), result)

        self.assertEqual(set(stored), {"nw-existing-v", personal_name})
        self.assertEqual(stored["nw-existing-v"], saved)
        self.assertTrue(all(w["name"] == personal_name for w in writes))
        if exists:
            self.assertTrue(all(w["id"] == "personal-id" for w in writes))
        spec = stored[personal_name]["spec"]["section"]
        self.assertEqual(spec["runSets"][0]["filters"]["filters"], [])
        self.assertEqual(spec["runSets"][0]["sort"]["keys"],
                         [{"key": {"section": "run", "name": "createdAt"}, "ascending": False}])
        feed = spec["runSets"][0]["runFeed"]
        visible = {key for key,value in feed["columnVisible"].items() if value}
        pinned = {key for key,value in feed["columnPinned"].items() if value}
        self.assertEqual(len(visible),17)
        self.assertNotIn("config:runtime.algorithm",visible)
        self.assertNotIn("summary:training_mode",visible)
        self.assertIn("summary:evaluation/episodes_completed",visible)
        self.assertTrue({"run:createdAt","run:duration","summary:eval/mean_delivery_rate_percent",
                         "summary:eval/success_rate"} <= visible)
        self.assertEqual(pinned,{"run:displayName","run:state","run:createdAt","run:duration"})
        plots = [p for section in spec["panelBankConfig"]["sections"] for p in section["panels"]
                 if "metrics" in p.get("config",{})]
        self.assertEqual(len(plots),37)
        paper_return = [p for p in plots if p.get("config", {}).get("chartTitle") == "UAV-paper training return / reward"]
        self.assertEqual(len(paper_return), 1)
        paper_config = paper_return[0]["config"]
        self.assertEqual(paper_config["xAxis"], "episode/index")
        self.assertEqual(paper_config["xExpression"], "${episode/index} / 100")
        self.assertEqual(paper_config["xAxisTitle"], "Episode (×10²)")
        self.assertEqual(paper_config["smoothingType"], "exponential")
        self.assertEqual(paper_config["smoothingWeight"], 0.99)
        self.assertEqual(paper_config["pointVisualizationMethod"], "sampling")
        self.assertFalse(paper_config["showOriginalAfterSmoothing"])


        success = [p for p in plots if p.get("config", {}).get("chartTitle") == "UAV-paper mission success (%)"]
        self.assertEqual(len(success), 1)
        success_config = success[0]["config"]
        self.assertEqual(success_config["metrics"], [])
        self.assertEqual(success_config["expressions"], ["${outcome/success} * 100"])
        self.assertEqual(success_config["xAxis"], "episode/index")
        self.assertEqual(success_config["xExpression"], "${episode/index} / 100")
        self.assertEqual(success_config["xAxisTitle"], "Episode (×10²)")
        self.assertEqual(success_config["yAxisTitle"], "Mission success (%)")
        self.assertEqual(success_config["yAxisMin"], 0.0)
        self.assertEqual(success_config["yAxisMax"], 100.0)

        rolling_success = [p for p in plots if p.get("config", {}).get("chartTitle") == "Rolling mission success rate (%) - trend"]
        self.assertEqual(len(rolling_success), 1)
        rolling_config = rolling_success[0]["config"]
        self.assertEqual(rolling_config["metrics"], [])
        self.assertEqual(rolling_config["expressions"], ["${rolling/success_rate} * 100"])
        self.assertEqual(rolling_config["xAxis"], "global_step")
        self.assertEqual(rolling_config["xExpression"], "${global_step} / 1000000")
        self.assertEqual(rolling_config["yAxisMin"], 0.0)
        self.assertEqual(rolling_config["yAxisMax"], 100.0)

        sample_return = [p for p in plots if p.get("config", {}).get("chartTitle") == "Sample-efficiency training return / reward"]
        self.assertEqual(len(sample_return), 1)
        self.assertEqual(sample_return[0]["config"]["xAxis"], "global_step")
        self.assertEqual(sample_return[0]["config"]["xExpression"], "${global_step} / 1000000")
        self.assertEqual(sample_return[0]["config"]["xAxisTitle"], "Environment Steps (M)")

        compute_return = [p for p in plots if p.get("config", {}).get("chartTitle") == "Compute-efficiency training return / reward"]
        self.assertEqual(len(compute_return), 1)
        self.assertEqual(compute_return[0]["config"]["xAxis"], "_runtime")
        self.assertEqual(compute_return[0]["config"]["xExpression"], "${_runtime} / 60")
        self.assertEqual(compute_return[0]["config"]["xAxisTitle"], "Wall Time (min)")
        for panel in plots:
            config = panel["config"]
            self.assertEqual(config.get("limit"),12)
            self.assertFalse(config.get("aggregate",False))
            metrics=config.get("metrics", [])
            if not metrics:
                continue
            metric=metrics[0]
            if metric.startswith("system/"): self.assertEqual(config["xAxis"],"_runtime")
            elif metric in {"evaluation/delivery_rate_percent","evaluation/success"}:
                self.assertEqual(config["xAxis"],"evaluation/episodes_completed")
        self.assertGreaterEqual(spec["settings"]["maxRuns"], 1000)
        self.assertEqual(spec["settings"]["xAxis"], "global_step")
        self.assertFalse(spec["settings"]["shouldAutoGeneratePanels"])
        self.assertEqual(sum(len(s["panels"]) for s in spec["panelBankConfig"]["sections"]),
                         sum(len(items) for items in PANELS.values()) + 1)

    def test_updates_existing_personal_workspace_without_creating_saved_views(self):
        self.check_personal_workspace(exists=True)

    def test_first_setup_and_repeated_setup_reuse_the_same_personal_workspace(self):
        self.check_personal_workspace(exists=False)

    def test_run_url_is_rejected_instead_of_overwriting_personal_workspace(self):
        responses = [
            {"viewer": {"username": "researcher"}},
            {"project": {"allViews": {"edges": []}}},
            {"project": {"runs": {"edges": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}},
            {"upsertView": {"view": {"id": "personal-id", "name": "nw-nwuserresearcher-w"}}},
        ]
        with TemporaryDirectory() as directory, chdir(directory):
            config = Config()
            config.tracking.credentials_file = str(Path(directory) / "absent-key.txt")
            with patch.dict(os.environ, {"WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
                    patch("uav_network.experiments.dashboard._graphql", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "Workspace URL"):
                    create_dashboard(config, workspace_url="https://wandb.ai/owner/project/runs/run123")


class DefaultVisibilityTests(unittest.TestCase):
    def test_reconcile_clears_hidden_runs_and_preserves_charts(self):
        from uav_network.experiments import dashboard
        self.assertTrue(callable(getattr(dashboard, "ensure_project_workspace", None)),
                        "default workspace visibility must be reconciled when runs start")
        old = {"section": {"panelBankConfig": {"custom": [1, 2]}, "settings": {"maxRuns": 2},
                "runSets": [{"filters": {"filterFormat": "filterV2", "filters": [{"value": "old-group"}]},
                             "search": {"query": "old-run"}, "selections": {"root": 0}, "enabled": False}]}}
        stored = {"id": "personal", "name": "nw-nwuserresearcher-w", "displayName": "Mine", "spec": json.dumps(old)}
        def query(document, variables=None):
            if "viewer" in document: return {"viewer": {"username": "researcher"}}
            if "allViews" in document: return {"project": {"allViews": {"edges": [{"node": stored}]}}}
            if "upsertView" in document:
                stored["spec"] = variables["input"]["spec"]
                return {"upsertView": {"view": {"id": stored["id"], "name": stored["name"]}}}
            raise AssertionError(document)
        c = Config()
        with patch.object(dashboard, "_authentication", return_value=("owner", "project")), \
                patch.object(dashboard, "_graphql", side_effect=query):
            dashboard.ensure_project_workspace(c)
        current = json.loads(stored["spec"])["section"]
        self.assertEqual(current["panelBankConfig"], old["section"]["panelBankConfig"])
        self.assertEqual(current["runSets"][0]["filters"]["filters"], [])
        self.assertEqual(current["runSets"][0]["search"]["query"], "")
        self.assertEqual(current["runSets"][0]["selections"]["root"], 1)
        self.assertTrue(current["runSets"][0]["enabled"])
        self.assertEqual(current["runSets"][0]["sort"]["keys"],
                         [{"key": {"section": "run", "name": "createdAt"}, "ascending": False}])
        self.assertGreaterEqual(current["settings"]["maxRuns"], 1000)
