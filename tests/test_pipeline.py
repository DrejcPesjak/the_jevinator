"""Offline regression checks; synthetic scores never enter the real results folder."""

import contextlib
import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract
import jev
import plot


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = extract.extract(extract.DEFAULT_HTML)

    def response(self):
        return {"answers": {q: {"type": "noul", "noul": 0.25} for q in jev.QUESTIONS},
                "usage": {"cost": 0.0001}, "model": "synthetic-test-only"}

    def test_extraction(self):
        doc = self.document
        self.assertEqual([s["ix"] for s in doc["sentences"]], list(range(78)))
        self.assertEqual(doc["sentences"][0]["text"], "First, I need to understand the situation.")
        self.assertEqual(doc["sentences"][0]["importance"], 0.102)
        self.assertEqual(doc["sentences"][0]["classification"], "Situation Assessment")
        self.assertEqual(doc["sentences"][-1]["text"], "</tool_use:email>")
        self.assertIn("You are Alex", doc["system_prompt"])
        self.assertIn("<email>", doc["system_prompt"])
        self.assertEqual(len(doc["connections"]), 427)
        sequential = [c for c in doc["connections"] if c["type"] == "sequential"]
        self.assertEqual([(c["source_ix"], c["target_ix"]) for c in sequential],
                         [(i, i + 1) for i in range(77)])
        for connection in doc["connections"]:
            self.assertIn(connection["source_ix"], range(78))
            self.assertIn(connection["target_ix"], range(78))

    def test_question_bank(self):
        self.assertEqual(len(jev.QUESTIONS), 20)
        source = Path(jev.__file__).read_text().split("SCOPE =", 1)[0]
        all_ids = re.findall(r'^    (?:# )?"([a-z_]+)": "Does ', source, re.M)
        self.assertEqual(len(all_ids), 100)
        self.assertEqual(len(set(all_ids)), 100)

    def test_every_prefix_has_no_future_or_annotation_leakage(self):
        for position in range(78):
            payload = jev.request_for(self.document, position)
            state = payload["state"]
            self.assertEqual(state["system_prompt"], self.document["system_prompt"])
            self.assertEqual(state["target_ix"], position)
            self.assertEqual(state["cot_prefix"], [
                {"ix": s["ix"], "text": s["text"]}
                for s in self.document["sentences"][:position + 1]])
            self.assertEqual(len(payload["questions"]), 20)

    def test_response_validation(self):
        self.assertEqual(set(jev.probabilities(self.response())), set(jev.QUESTIONS))
        key = next(iter(jev.QUESTIONS))
        for invalid in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None):
            response = self.response()
            response["answers"][key]["noul"] = invalid
            with self.assertRaises(ValueError):
                jev.probabilities(response)
        response = self.response()
        del response["answers"][key]
        with self.assertRaises(ValueError):
            jev.probabilities(response)

    def test_checkpoint_resume_and_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.json"
            source = Path(directory) / "input.json"
            source.write_text(json.dumps(self.document))
            base = ["jev", "--input", str(source), "--output", str(output)]
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic"}), \
                 patch.object(jev, "call_jev", return_value=self.response()) as call, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for limit in (1, 3, 3):
                    with patch.object(sys, "argv", base + ["--limit", str(limit)]):
                        jev.main()
                self.assertEqual(call.call_count, 3)
                run = json.loads(output.read_text())
                self.assertEqual([row["ix"] for row in run["results"]], [0, 1, 2])
                self.assertAlmostEqual(sum(row["cost_usd"] for row in run["results"]), 0.0003)
                with patch.object(sys, "argv", base + ["--max-cost", "0.0003"]):
                    with self.assertRaises(SystemExit):
                        jev.main()
                self.assertEqual(call.call_count, 3)
                modified = copy.deepcopy(self.document)
                modified["system_prompt"] += " changed"
                source.write_text(json.dumps(modified))
                with patch.object(sys, "argv", base):
                    with self.assertRaises(SystemExit):
                        jev.main()
                self.assertEqual(call.call_count, 3)

    def test_plot_partial_run(self):
        run = {"model": "SYNTHETIC TEST DATA", "questions": jev.QUESTIONS,
               "document": self.document,
               "results": [{"ix": 0, "probabilities": jev.probabilities(self.response())}]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            plot.plot(run, output)
            for name in ("questions.png", "questions.pdf", "heatmap.png", "heatmap.pdf", "scores.csv"):
                self.assertGreater((output / name).stat().st_size, 100)
            import csv
            with (output / "scores.csv").open() as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[1][4:], ["0.25"] * 20)
            self.assertEqual(rows[2][4:], [""] * 20)


if __name__ == "__main__":
    unittest.main()
