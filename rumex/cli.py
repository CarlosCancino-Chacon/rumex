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

import argparse
import json
import os

import numpy as np
import partitura

from rumex.render import render_piece, load_score, DEFAULT_CONFIG


def remove_score_extension(filename):
    for ex in (".musicxml", ".xml", ".mei", ".midi", ".mid"):
        if filename.lower().endswith(ex):
            return filename[: -len(ex)]
    return filename


def main():

    parser = argparse.ArgumentParser("Generate an expressive performance from a piece")

    parser.add_argument(
        "filename", help="Score to be rendered (MusicXML or MIDI)"
    )
    parser.add_argument("--config", help="Config file in JSON format", default=None)
    parser.add_argument(
        "--return-bfs",
        "--return-features",
        dest="return_features",
        help="Save the note features and the expressive parameters (.npz)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--outfile", help="Output MIDI file (optional)", default=None
    )
    parser.add_argument(
        "--random-state",
        help="Seed for random number generator",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--features-outfile",
        help="Output file for storing the note features and parameters",
        default=None,
    )
    parser.add_argument(
        "--save-match", help="Save match file", action="store_true", default=False
    )

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG

    out_dir = os.path.dirname(args.filename)
    basename = remove_score_extension(os.path.basename(args.filename))

    if args.outfile is None:
        args.outfile = os.path.join(out_dir, basename + ".mid")

    if args.features_outfile is None:
        args.features_outfile = os.path.join(out_dir, basename + ".npz")

    rng = np.random.RandomState(args.random_state)

    part = load_score(args.filename)

    # render piece
    predicted_ppart, W, names, predictions, alignment = render_piece(
        part, config=config, rng=rng, return_features=True
    )

    if args.return_features:
        results = dict(
            W=W,
            bf_names=names,
            predictions=predictions,
            score_info=part.note_array(),
        )
        np.savez_compressed(args.features_outfile, **results)

    if args.save_match:
        out_match = os.path.splitext(args.outfile)[0] + ".match"
        partitura.save_match(
            alignment=alignment,
            performance_data=predicted_ppart,
            score_data=part,
            out=out_match,
            performer="RuMEx",
            piece=basename,
        )
    # save as MIDI file
    partitura.save_performance_midi(predicted_ppart, args.outfile)


if __name__ == "__main__":
    main()
