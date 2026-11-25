# Copyright (C) 2020-2026 Carlos Cancino-Chacón
#
# This file is part of RuMEx.
#
# RuMEx is free software: you can redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# RuMEx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License along with RuMEx. If not, see
# <https://www.gnu.org/licenses/>.

import os

import numpy as np
import partitura

from rumex import (
    compute_note_features,
    load_score,
    render_piece,
    DEFAULT_CONFIG,
    double_notes_keep_mask,
)

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_examples",
    "mozart_k331.musicxml",
)


def test_note_features():
    part = load_score(EXAMPLE)
    W, names = compute_note_features(part)
    assert W.shape == (len(part.note_array()), len(names))
    for name in ("loudness_incr", "tempo_incr", "beat_phase", "fermata", "slur_incr"):
        assert name in names


def test_render_piece(tmp_path):
    part = load_score(EXAMPLE)
    ppart, W, names, predictions, alignment = render_piece(
        part, rng=np.random.RandomState(1), return_features=True
    )
    keep = double_notes_keep_mask(part)
    assert len(ppart.notes) == keep.sum() == len(alignment)
    assert np.all(np.isfinite([n["note_on"] for n in ppart.notes]))
    assert all(1 <= n["velocity"] <= 127 for n in ppart.notes)
    # same seed, same performance
    ppart2 = render_piece(part, rng=np.random.RandomState(1))
    assert [n["note_on"] for n in ppart.notes] == [n["note_on"] for n in ppart2.notes]
    # the performance can be saved
    partitura.save_performance_midi(ppart, str(tmp_path / "out.mid"))


def _profile(increase, decrease=None):
    from rumex.base import compute_increase_decrease_profile

    onsets = np.arange(7, dtype=float)
    markings = lambda o: 0 if o < 3 else 1  # noqa: E731  (a new marking starts at onset 3)
    cm_array = np.array([[0.0, 0], [3.0, 1]])
    if decrease is None:
        decrease = np.zeros(len(onsets))
    return compute_increase_decrease_profile(
        np.array(increase, dtype=float), decrease, onsets, markings, cm_array,
        parameter_per_marking=[40, 60], default_diff=10,
    )


def test_ramp_ending_at_new_marking():
    # partitura ramps are 1 at the end of the direction: here the crescendo ends where the next marking starts
    profile = _profile([0, 1 / 3, 2 / 3, 1, 0, 0, 0])
    # the next section is not changed by the end point of the ramp
    assert np.all(profile[3:] == 0)
    # the crescendo reaches 90% of the difference between the two markings at the last onset before the end
    assert np.isclose(profile[2], 0.9 * 20)


def test_ramp_ending_inside_section():
    # a crescendo that ends inside a section keeps its value until the next marking
    profile = _profile([0, 0, 1 / 3, 2 / 3, 1, 0, 0])
    assert profile[4] > profile[3] > 0
    assert profile[5] == profile[6] == profile[4]


def _tempo_markings():
    from rumex.tempo import CONSTANT_MARKINGS

    idx = {n: i for i, n in reversed(list(enumerate(CONSTANT_MARKINGS)))}
    names = ["allegro", "moderato", "allegro", "allegro", "moderato"]
    return [(10.0 * k, idx[n]) for k, n in enumerate(names)], names


def test_same_tempo_for_same_marking():
    from rumex.tempo import _sample_beat_period_per_tm

    tm_array, names = _tempo_markings()
    for seed in range(5):
        bp = np.array(_sample_beat_period_per_tm(tm_array, np.random.RandomState(seed)))
        assert np.all(bp[[0, 2, 3]] == bp[0]) and bp[1] == bp[4]
        assert bp[0] != bp[1]


def test_tempo_variation_within_range():
    from rumex.tempo import _sample_beat_period_per_tm

    tm_array, names = _tempo_markings()
    variation = 0.03
    for seed in range(20):
        bpm = 60 / np.array(
            _sample_beat_period_per_tm(tm_array, np.random.RandomState(seed), variation)
        )
        # instances of the same marking differ by at most 2 * variation (relative to the tempo of the marking)
        for group in ([0, 2, 3], [1, 4]):
            assert bpm[group].max() / bpm[group].min() <= (1 + variation) / (1 - variation) + 1e-9
        assert len(set(bpm[[0, 2, 3]])) > 1


def test_tempo_variation_pipeline():
    from rumex import DEFAULT_CONFIG

    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    pipeline = [
        "tempo_markings_variation_rules" if r == "tempo_markings_rules" else r
        for r in config["tempo"]["pipeline"]
    ]
    assert pipeline != config["tempo"]["pipeline"]
    config["tempo"]["pipeline"] = pipeline
    part = load_score(EXAMPLE)
    ppart = render_piece(part, config=config, rng=np.random.RandomState(1))
    assert len(ppart.notes) == double_notes_keep_mask(part).sum()


def test_pipeline_callables_and_melody_loudness():
    from rumex import DynamicsRules, DEFAULT_DYNAMICS_PIPELINE
    from rumex.dynamics import skyline

    part = load_score(EXAMPLE)
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}

    def velocities(r):
        cfg["dynamics"]["pipeline"] = DEFAULT_DYNAMICS_PIPELINE + [
            ("melody_loudness_rules", {"melody_relative_loudness": r})
        ]
        pp = render_piece(part, config=cfg, rng=np.random.RandomState(3))
        by_id = {n["id"]: n["velocity"] for n in pp.notes}
        return np.array([by_id[i] for i in na["id"]])  # in the order of the notes of the part

    from rumex import expanded_note_array

    # the rules see the grace notes expanded and no double notes
    na = expanded_note_array(part)[double_notes_keep_mask(part)]
    onsets = na["onset_beat"]
    uniq = np.unique(onsets)
    groups = [np.where(onsets == u)[0] for u in uniq]
    melody = skyline(na["pitch"], groups)
    assert melody.sum() == len(uniq)

    v0, v1, vm = velocities(0.0), velocities(0.5), velocities(-0.5)
    assert np.all(v1[~melody] == v0[~melody]) and np.all(vm[~melody] == v0[~melody])
    assert np.all(v1[melody] >= v0[melody]) and np.all(vm[melody] <= v0[melody])
    assert v1[melody].mean() > v0[melody].mean() > vm[melody].mean()
    assert v1.max() <= 127 and vm.min() >= 1

    # a user-defined callable works as well
    def my_rule(velocity_trend, velocity_dev, *args, **kwargs):
        return velocity_trend, velocity_dev

    DynamicsRules(pipeline=["dynamics_markings_rules", my_rule])
    try:
        DynamicsRules(pipeline=["no_such_rule"])
        assert False
    except ValueError:
        pass


def test_monophonic_skyline():
    from rumex.dynamics import skyline, monophonic_skyline
    from partitura.musicanalysis.performance_codec import get_unique_onset_idxs

    # a held melody note (72) over a moving accompaniment, then a higher note (74) while it is held
    pitch = np.array([72, 60, 64, 67, 74])
    onsets = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    duration = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
    basic = skyline(pitch, get_unique_onset_idxs(onsets))
    assert basic.tolist() == [True, False, True, True, True]  # overlapping notes
    mono = monophonic_skyline(pitch, onsets, duration)
    assert mono.tolist() == [True, False, False, False, False]
    # after the melody note ended, the highest note of the next onset is selected
    mono = monophonic_skyline(
        np.array([72, 60, 55, 70, 50]),
        np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert mono.tolist() == [True, False, True, True, False]
    # after a rest, the highest note is the melody
    mono = monophonic_skyline(np.array([72, 60, 55]), np.array([0.0, 2.0, 2.0]), np.ones(3))
    assert mono.tolist() == [True, True, False]
    # on a real piece the melody never overlaps itself
    part = load_score(EXAMPLE)
    na = part.note_array()
    m = monophonic_skyline(na["pitch"], na["onset_beat"], na["duration_beat"])
    on = na["onset_beat"][m]
    en = on + na["duration_beat"][m]
    assert np.all(en[:-1] <= on[1:] + 1e-3)


GRACE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><grace slash="yes"/><pitch><step>D</step><octave>5</octave></pitch><type>eighth</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><grace/><pitch><step>F</step><octave>5</octave></pitch><type>eighth</type></note>
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
"""


def test_grace_notes(tmp_path):
    fn = tmp_path / "grace.musicxml"
    fn.write_text(GRACE_XML)
    part = load_score(str(fn))
    assert len(part.note_array()) == 10
    assert len(load_score(str(fn), grace_notes="remove").note_array()) == 8

    from rumex import expanded_note_array

    na = part.note_array()
    ex = expanded_note_array(part)
    assert (ex["id"] == na["id"]).all()  # same order as the part
    # acciaccatura (slashed): before the beat, takes time from the previous note
    acc = np.where((na["pitch"] == 74))[0][0]
    assert ex["onset_beat"][acc] < na["onset_beat"][acc] and ex["duration_beat"][acc] > 0
    prev = np.where(na["pitch"] == 72)[0][0]
    assert ex["duration_beat"][prev] < na["duration_beat"][prev]
    assert abs(ex["onset_beat"][acc] + ex["duration_beat"][acc] - na["onset_beat"][acc]) < 1e-6

    ppart, W, names, predictions, alignment = render_piece(
        part, rng=np.random.RandomState(0), return_features=True
    )
    assert len(ppart.notes) == len(alignment) == 10
    by_id = {n["id"]: n for n in ppart.notes}
    # the ids of the performed notes have the pitch of their score notes
    for a in alignment:
        i = list(na["id"]).index(a["score_id"])
        assert by_id[a["performance_id"]]["midi_pitch"] == na["pitch"][i]
    # the grace note is played before the main note
    main = by_id[na["id"][np.where(na["pitch"] == 76)[0][0]]]
    assert by_id[na["id"][acc]]["note_on"] < main["note_on"]
    # without the expansion the grace note has no duration
    pp0 = render_piece(part, rng=np.random.RandomState(0), expand_grace_notes=False)
    assert len(pp0.notes) == 10


DOUBLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>half</type></note>
      <backup><duration>8</duration></backup>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration><voice>2</voice><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>2</voice><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>2</voice><type>half</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>4</duration><type>half</type></note>
      <note><pitch><step>B</step><octave>4</octave></pitch><duration>4</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""


def test_double_notes(tmp_path):
    fn = tmp_path / "double.musicxml"
    fn.write_text(DOUBLE_XML)
    part = load_score(str(fn))
    na = part.note_array()
    assert len(na) == 7
    # the C5 (half, voice 1) and the C5 (quarter, voice 2) have the same onset
    longer = double_notes_keep_mask(part)
    assert longer.sum() == 6
    c5 = np.where(na["pitch"] == 72)[0]
    assert longer[c5].sum() == 1 and na["duration_beat"][c5][longer[c5]][0] == 2
    shorter = double_notes_keep_mask(part, keep_longer=False)
    assert shorter.sum() == 6 and na["duration_beat"][c5][shorter[c5]][0] == 1

    ppart, W, names, predictions, alignment = render_piece(
        part, rng=np.random.RandomState(0), return_features=True
    )
    assert len(ppart.notes) == len(alignment) == len(W) == len(predictions) == 6
    on = {}
    for n in ppart.notes:  # no two notes with the same pitch start together
        key = (n["midi_pitch"], round(n["note_on"], 3))
        assert key not in on
        on[key] = 1
    assert len(render_piece(part, remove_double_notes=False).notes) == 7
