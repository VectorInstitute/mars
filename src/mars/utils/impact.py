"""Energy, carbon and water of a run, measured with ai-impact-accounting.

Each stage is measured on the main rank only: the meters read every GPU on the
node, so measuring on every rank would count the node once per rank. Training and
the source-validation pass go to the run's impact.json; each scored corpus
(stage eval/<corpus>) goes to <corpus>/impact.json, next to its predictions.
Reports are the dia_report blocks track() produces, so they can be pushed to a
model card later without measuring again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_impact_accounting import track

from mars.utils.distributed import DistributedContext
from mars.utils.logging import get_logger


class Footprint:
    """Measure named stages of a run and write them to impact.json.

    Parameters
    ----------
    base_model : str
        Pretrained model the method builds on, recorded as lineage.
    relation : str
        How the method relates to it, e.g. adapter or finetune.
    ctx : DistributedContext
        Only the main rank measures.
    """

    def __init__(self, base_model: str, relation: str, ctx: DistributedContext) -> None:
        self.base_model = base_model
        self.relation = relation
        self.is_main = ctx.is_main
        self.meters: dict[str, track] = {}

    def start(self, stage: str) -> None:
        """Start measuring stage on the main rank."""
        if self.is_main:
            self.meters[stage] = track(base_model=self.base_model, relation=self.relation).__enter__()

    def stop(self, stage: str) -> None:
        """Stop measuring stage and log its one-line summary."""
        if stage in self.meters:
            meter = self.meters[stage]
            meter.__exit__(None, None, None)
            get_logger().info("%s: %s", stage, meter.checklist_line())

    def metrics(self) -> dict[str, float]:
        """Flat footprint scalars per stage, for the run's metric table."""
        row: dict[str, float] = {}
        for stage, meter in self.meters.items():
            low, high = meter.water_l
            row |= {
                f"impact/{stage}/gpu_hours": meter.gpu_hours,
                f"impact/{stage}/energy_kwh": meter.energy_kwh or 0.0,
                f"impact/{stage}/carbon_kg": meter.carbon_kg or 0.0,
                f"impact/{stage}/water_l_low": low,
                f"impact/{stage}/water_l_high": high,
            }
        return row

    def write(self, results_dir: str | Path) -> None:
        """Write each corpus's report to its folder; merge the rest into impact.json."""
        if not self.meters:
            return
        root = Path(results_dir)
        run: dict[str, Any] = {}
        for stage, meter in self.meters.items():
            corpus = stage.removeprefix("eval/")
            if corpus == stage:
                run[stage] = meter.report_dict()
                continue
            path = root / corpus / "impact.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meter.report_dict(), indent=2))
        if run:
            path = root / "impact.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            reports: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
            path.write_text(json.dumps(reports | run, indent=2))
