"""The trained Gate 1 head (gate1.decider: head) with fake probabilities and a tiny head file."""
import numpy as np
import pytest

from app.core.config import Gate1Settings, settings
from app.gates.gate1 import run_gate1
from app.ml import head as head_module
from app.ml.head import Gate1Head
from app.pipeline import orchestrator
from tests.test_gate1 import DOG, FOOD, LABELS, SCALE, make_ctx, row
from tests.test_gate2_pipeline import PERSONAL, with_model  # noqa: F401


@pytest.fixture
def head_decides(monkeypatch):
    monkeypatch.setattr(settings.gate1, "decider", "head")
    monkeypatch.setattr(settings.gate1.head, "pass_from", 0.5)
    monkeypatch.setattr(settings.gate1.head, "fail_below", 0.2)
    monkeypatch.setattr(settings.gate1.head, "wild_from", 0.5)


def judged(probs, values=DOG, is_video=False):
    sims = np.stack([row(**values)] * len(probs))
    return run_gate1(sims, LABELS, SCALE, is_video, head=probs)


# --- decisions from the head's probabilities ---

def test_head_sure_of_a_pet_passes(head_decides):
    result = judged([{"pet": 0.85, "wild": 0.05, "no_pet": 0.10}])
    assert result.outcome == "pass" and not result.wild
    assert result.pet_score == 90
    assert result.threshold == 50


def test_head_overrules_the_rules_on_a_lookalike(head_decides):
    # SigLIP calls the fur coat "a dog" (rules: pass); the head learnt fur coats are no pet
    result = judged([{"pet": 0.05, "wild": 0.02, "no_pet": 0.93}], values=DOG)
    assert result.outcome == "fail"


def test_head_can_pass_what_the_rules_reject(head_decides):
    result = judged([{"pet": 0.8, "wild": 0.0, "no_pet": 0.2}], values=FOOD)
    assert result.outcome == "pass"


def test_unsure_head_sends_to_a_human(head_decides):
    assert judged([{"pet": 0.3, "wild": 0.05, "no_pet": 0.65}]).outcome == "borderline"


def test_wild_animal_passes_gate1_but_is_flagged_wild(head_decides):
    result = judged([{"pet": 0.2, "wild": 0.7, "no_pet": 0.1}])
    assert result.outcome == "pass" and result.wild


def test_video_counts_frames_the_head_passes(head_decides):
    pet, empty = {"pet": 0.9, "wild": 0.0, "no_pet": 0.1}, {"pet": 0.0, "wild": 0.0, "no_pet": 1.0}
    assert judged([pet, empty, empty], is_video=True).outcome == "pass"           # 33% >= 30%
    assert judged([pet, empty, empty, empty], is_video=True).outcome == "borderline"   # 25%


@pytest.fixture
def head_checks(monkeypatch):
    monkeypatch.setattr(settings.gate1, "decider", "check")
    monkeypatch.setattr(settings.gate1.head, "check_no_pet_from", 0.8)


def test_check_sends_a_rules_pass_the_head_calls_no_pet_to_a_human(head_checks):
    # the fur coat: SigLIP says "a dog" (rules pass), the head is 0.99 sure there is no pet
    result = judged([{"pet": 0.0, "wild": 0.01, "no_pet": 0.99}], values=DOG)
    assert result.outcome == "borderline" and result.head_doubt
    assert result.pet_score >= 75   # the rules' score is kept


def test_check_leaves_a_rules_pass_the_head_agrees_with(head_checks):
    result = judged([{"pet": 0.9, "wild": 0.0, "no_pet": 0.1}], values=DOG)
    assert result.outcome == "pass" and not result.head_doubt


def test_check_never_rescues_a_rules_reject(head_checks):
    result = judged([{"pet": 0.99, "wild": 0.0, "no_pet": 0.01}], values=FOOD)
    assert result.outcome == "fail"


def test_check_on_a_video_needs_most_passed_frames_doubted(head_checks):
    sure, doubt = {"pet": 0.9, "wild": 0.0, "no_pet": 0.1}, {"pet": 0.0, "wild": 0.0, "no_pet": 0.95}
    assert judged([sure, sure, doubt], is_video=True).outcome == "pass"
    assert judged([doubt, doubt, sure], is_video=True).outcome == "borderline"


def test_pipeline_check_sends_the_doubt_to_review(tmp_path, with_model, head_checks, monkeypatch):
    with_model(PERSONAL)
    monkeypatch.setattr(orchestrator, "get_gate1_head", lambda: FakeHead({"pet": 0.01, "wild": 0.0, "no_pet": 0.99}))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert "gate1_head_doubt" in response.flags


def test_rules_decide_when_no_head_is_given():
    result = run_gate1(row(**DOG)[None, :], LABELS, SCALE, is_video=False)
    assert result.frames[0].head is None and result.threshold == 75


def test_fail_bar_above_pass_bar_is_refused():
    data = settings.gate1.model_dump()
    data["head"]["fail_below"] = 0.6
    with pytest.raises(ValueError, match="fail_below must not be above pass_from"):
        Gate1Settings.model_validate(data)


# --- the head file ---

def save_head(path, siglip_model, weights=None):
    np.savez(
        path, weights=weights if weights is not None else np.zeros((3, 4), np.float32),
        bias=np.zeros(3, np.float32), classes=np.array(["pet", "wild", "no_pet"]),
        features=np.array(["siglip"]), siglip_model=np.array(siglip_model),
        bioclip_model=np.array("unused"), trained_on=np.array("test"),
    )


@pytest.fixture
def head_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.ml, "backbone", "siglip2_base")
    monkeypatch.setattr(head_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings.gate1.head, "file", "head.npz")
    return tmp_path / "head.npz"


def test_head_file_gives_softmax_probabilities(head_file):
    weights = np.zeros((3, 4), np.float32)
    weights[0, 0] = 10.0            # first number high -> pet
    save_head(head_file, f"siglip2_base@{settings.ml.models['siglip2_base'].revision}", weights)
    head = Gate1Head()
    head.load()

    [p] = head.probabilities({"siglip": np.array([[1.0, 0, 0, 0]], np.float32)})
    assert p["pet"] > 0.99 and abs(sum(p.values()) - 1) < 1e-6


def test_head_from_another_model_revision_is_refused(head_file):
    save_head(head_file, "siglip2_base@0000000")
    with pytest.raises(RuntimeError, match="head trained on"):
        Gate1Head().load()


# --- in the pipeline ---

class FakeHead:
    features = ["siglip"]

    def __init__(self, probs):
        self.probs = probs

    def probabilities(self, features):
        return [self.probs] * len(features["siglip"])


def test_pipeline_rejects_what_the_head_calls_no_pet(tmp_path, with_model, head_decides, monkeypatch):
    with_model(PERSONAL)   # the rules would approve this dog
    monkeypatch.setattr(orchestrator, "get_gate1_head", lambda: FakeHead({"pet": 0.02, "wild": 0.0, "no_pet": 0.98}))

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.rejection_code == "MOD_001"


def test_missing_head_falls_back_to_the_rules(tmp_path, with_model, head_decides, monkeypatch):
    with_model(PERSONAL)
    monkeypatch.setattr(orchestrator, "get_gate1_head", lambda: None)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert "gate1_head_unavailable" in response.flags
