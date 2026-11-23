# RuMEx

RuMEx is a rule-based system for rendering expressive performances.

## Setup
RuMEx only depends on [Partitura](https://github.com/CPJKU/partitura) (note features, performance codec, score and
performance I/O), numpy and scipy, which are installed automatically. From the repository folder:
```bash
pip install .
```
This also installs the `RuMExRender` command. For development use an editable install, `pip install -e .`
It requires Python 3.10 or newer, `partitura` 1.9 or newer and `numpy` 2 (developed and tested with Python 3.12, `partitura` 1.9.0, `numpy` 2.4 and `scipy` 1.18).

## Usage

The main program is `RuMExRender`. This program allows for generating an expressive performance given a score in MusicXML (preferred) or MIDI format.

The simplest way of generating a performance is
```bash
RuMExRender --outfile out.mid score_filename.xml
```

or, from Python,
```python
import numpy as np
import partitura
from rumex import load_score, render_piece

part = load_score("score_filename.xml")
ppart = render_piece(part, rng=np.random.RandomState(1984))
partitura.save_performance_midi(ppart, "out.mid")
```

For more information see 
```bash
RuMExRender -h
```

## Configuration

The rule pipelines and the note features that they use are set in a config (a JSON file for `RuMExRender --config`, or
a dict for `render_piece`; see `rumex.DEFAULT_CONFIG`). For each of `dynamics`, `tempo`, `timing` and `articulation`:

* `pipeline`: names of the rules (functions of the corresponding module),
* `note_features`: `"all"` or a list of the names of [partitura note feature
  functions](https://github.com/CPJKU/partitura/blob/main/partitura/musicanalysis/note_features.py), e.g.
  `["loudness_direction_feature", "slur_feature"]` (the old key `basis_functions` is also read).

The expressive parameters (`beat_period`, `velocity`, `timing`, `articulation_log`) are turned into a performance with
`partitura.musicanalysis.decode_performance`.

## License

Copyright (C) 2020-2026 Carlos Cancino-Chacón

RuMEx is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See the [LICENSE](LICENSE) file (GNU GPL version
3) for details.
