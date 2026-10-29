"""gate2-0.18: D1-D4 from Shieldstral policy questions (app.gates.guard), with a fake model."""
import pytest
from PIL import Image

from app.core.config import Gate2Settings, settings
from app.gates.guard import GuardResult, guard_frames, guard_signals, guard_views, pet_crop, run_guard, steady_review
from app.pipeline import orchestrator
from tests.test_gate1 import make_ctx
from tests.test_gate2_pipeline import PERSONAL, with_model  # noqa: F401


class FakeGuard:
    """Same score for every picture; a different one for the crop when `crop` is given."""

    def __init__(self, scores=None, crop=None):
        self.full, self.crop = scores or {}, crop
        self.calls = []

    def scores(self, images, questions, instruct):
        self.calls.append([img.size for img in images])
        out = []
        for i, _ in enumerate(images):
            values = self.crop if (i == 1 and self.crop is not None) else self.full
            out.append({name: values.get(name, 0.0) for name in questions})
        return out


@pytest.fixture
def guard_sees(monkeypatch):
    monkeypatch.setattr(settings.ml, "guard_model", "shieldstral")

    def use(scores=None, crop=None):
        fake = FakeGuard(scores, crop)
        monkeypatch.setattr(orchestrator, "get_guard_model", lambda: fake)
        return fake
    return use


# --- scores -> signals ---

def test_low_scores_give_no_signal():
    assert guard_signals({"distress": 0.15, "gore": 0.1}) == (set(), set())


def test_review_band_gives_the_review_signal():
    # the stressed husky at the groomer: distress 0.46 (review_from 0.25, act_from 0.5)
    assert guard_signals({"distress": 0.46}) == (set(), {"guard_concern_welfare"})


def test_from_act_from_the_question_signal_fires():
    assert guard_signals({"distress": 0.7}) == ({"welfare_distress"}, set())
    assert guard_signals({"fight": 0.9}) == ({"welfare_fighting"}, set())   # organised fighting: hard


def test_crate_with_puppies_stays_below_restrained_review():
    # safe photos: crates with a mother and babies reached 0.33; review_from is 0.35
    assert guard_signals({"restrained": 0.33}) == (set(), set())


# --- frames and crops ---

def test_image_asks_its_one_frame_video_spreads_frames():
    assert guard_frames(1, is_video=False) == [0]
    picks = guard_frames(61, is_video=True)
    assert picks[0] == 0 and picks[-1] == 60
    assert len(picks) == settings.gate2.guard.video_frames


def test_video_asks_the_most_suspicious_moments_and_the_next_frame():
    # a bite at 2.5 s and distress at 29 s fell between the blind 5-second steps
    suspicion = [0.01] * 61
    suspicion[2], suspicion[29], suspicion[45] = 0.8, 0.7, 0.6
    picks = guard_frames(61, is_video=True, suspicion=suspicion)
    assert {2, 3, 29, 30, 45, 46} <= set(picks)
    assert 0 in picks and 60 in picks              # the rest still spread over the video
    assert len(picks) <= settings.gate2.guard.video_frames + 1


def test_one_borderline_video_frame_is_noise_two_are_not():
    once = [GuardResult(signals={"guard_concern_violence"}, review={"guard_concern_violence"}),
            GuardResult(), GuardResult()]
    assert all(not r.signals for r in steady_review(once))
    twice = [GuardResult(signals={"guard_concern_welfare"}, review={"guard_concern_welfare"}),
             GuardResult(signals={"guard_concern_welfare"}, review={"guard_concern_welfare"})]
    assert all(r.signals == {"guard_concern_welfare"} for r in steady_review(twice))


def test_a_strong_answer_counts_from_one_frame():
    results = [GuardResult(signals={"welfare_abuse"}), GuardResult(), GuardResult()]
    assert steady_review(results)[0].signals == {"welfare_abuse"}


def test_crop_grows_the_pet_box_to_take_in_the_handler():
    picture = Image.new("RGB", (640, 480))
    crop = pet_crop(picture, (300, 200, 400, 300))      # 100 x 100 box, margin 0.5 -> 200 x 200
    assert crop.size == (200, 200)


def test_no_crop_when_the_pet_fills_the_picture():
    picture = Image.new("RGB", (640, 480))
    assert pet_crop(picture, (50, 50, 600, 450)) is None
    assert pet_crop(picture, None) is None


def test_views_are_full_frame_plus_crop_at_most_image_size():
    picture = Image.new("RGB", (1280, 960))
    views = guard_views(picture, (500, 400, 700, 600))
    assert len(views) == 2
    assert max(views[0].size) == settings.gate2.guard.image_size


def test_the_higher_of_full_frame_and_crop_counts():
    # the bite moment: full frame hurt 0.05, crop around the dog and groomer 0.42
    [result] = run_guard(FakeGuard({"hurt": 0.05}, crop={"hurt": 0.42}),
                         [Image.new("RGB", (640, 480))], [(300, 200, 400, 300)])
    assert result.scores["hurt"] == 0.42
    assert result.signals == {"guard_concern_welfare"}


# --- settings checks ---

def test_question_with_unknown_signal_is_refused():
    data = settings.gate2.model_dump()
    data["guard"]["questions"]["hurt"]["signal"] = "welfare_kicking"
    with pytest.raises(ValueError, match="unknown signals"):
        Gate2Settings.model_validate(data)


def test_review_signal_must_always_reach_a_human():
    data = settings.gate2.model_dump()
    data["guard"]["questions"]["hurt"]["review_signal"] = "welfare_malnourished"   # not review
    with pytest.raises(ValueError, match="review_signal must be a review signal"):
        Gate2Settings.model_validate(data)


def test_review_from_must_be_below_act_from():
    data = settings.gate2.model_dump()
    data["guard"]["questions"]["hurt"]["review_from"] = 0.6
    with pytest.raises(ValueError, match="review_from must be below act_from"):
        Gate2Settings.model_validate(data)


# --- in the pipeline ---

def test_clean_answers_keep_the_approval(tmp_path, with_model, guard_sees):
    with_model(PERSONAL)
    fake = guard_sees({"distress": 0.02})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert fake.calls   # the guard was asked


def test_review_band_sends_the_post_to_a_human(tmp_path, with_model, guard_sees):
    with_model(PERSONAL)
    guard_sees({"distress": 0.46})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.dimension_scores.D1_welfare == 90


def test_person_hurting_the_pet_is_a_hard_reject(tmp_path, with_model, guard_sees):
    with_model(PERSONAL)
    guard_sees({"hurt": 0.8})

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "REJECTED"
    assert response.hard_reject is True
    assert response.rejection_code == "MOD_002"


def test_raw_feeding_softens_gore_but_a_human_checks(tmp_path, with_model, guard_sees):
    with_model({**PERSONAL, "scene_context__raw_feeding": 0.3})
    guard_sees({"gore": 0.63})   # gloved hand holding raw meat to a cat (safe set)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert response.dimension_scores.D3_violence == 75   # spec 7: -25 instead of -60


def test_guard_that_failed_to_load_never_approves(tmp_path, with_model, monkeypatch):
    with_model(PERSONAL)
    monkeypatch.setattr(settings.ml, "guard_model", "shieldstral")
    monkeypatch.setattr(orchestrator, "get_guard_model", lambda: None)

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "HUMAN_REVIEW"
    assert "guard_unavailable" in response.flags


def test_guard_off_changes_nothing(tmp_path, with_model):
    with_model(PERSONAL)   # ml.guard_model is null on the laptop

    response = orchestrator.run_pipeline(make_ctx(tmp_path))

    assert response.decision == "APPROVED"
    assert "guard_unavailable" not in response.flags
