from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from marketsim.LLM.reflection_pipeline.diagnosis_agreement import (
    _feature_present,
    _form_class_shape_ok,
    _parse_features_used,
    _principle_hint_ok,
    validate_code_implements_diagnosis,
)
from marketsim.LLM.reflection_pipeline.prompts import parse_principle_form_fields


@dataclass(frozen=True)
class HypothesisAdherence:
    principle: Optional[str]
    form_class: Optional[str]
    principle_ok: bool
    form_class_ok: bool
    feature_coverage: float
    overall: float
    violation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)




def score_hypothesis_adherence(
    code: str,
    reflection: str,
    *,
    strategy_type: str = "pv",
) -> HypothesisAdherence:
    fields = parse_principle_form_fields(reflection or "")
    principle = fields.get("principle")
    form_class = fields.get("form_class")
    if not principle or not form_class:
        return HypothesisAdherence(
            principle=principle,
            form_class=form_class,
            principle_ok=True,
            form_class_ok=True,
            feature_coverage=1.0,
            overall=1.0,
            violation=None,
        )

    violation = validate_code_implements_diagnosis(
        code,
        reflection,
        strategy_type=strategy_type,
    )
    principle_ok = _principle_hint_ok(principle, code)
    form_ok = _form_class_shape_ok(form_class, code)

    features = [
        f
        for f in _parse_features_used(reflection or "")
        if f
    ]
    if features:
        matched = sum(1 for feat in features if _feature_present(code, feat))
        feature_coverage = matched / float(len(features))
    else:
        feature_coverage = 1.0

    components = [float(principle_ok), float(form_ok), feature_coverage]
    overall = sum(components) / len(components)

    return HypothesisAdherence(
        principle=principle,
        form_class=form_class,
        principle_ok=principle_ok,
        form_class_ok=form_ok,
        feature_coverage=feature_coverage,
        overall=overall,
        violation=violation,
    )
