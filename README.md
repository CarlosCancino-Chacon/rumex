# RuMEx

RuMEx is a rule-based system for rendering expressive performances.

## Setup
It is important to have the Partitura and Basis Mixer packages set up in your environment.

* [Partitura](https://github.com/OFAI/partitura): Is best to install from GitHub and use the `develop` branch (the version in PyPI is not yet up-to-date and has a few bugs), for example
```
pip install git+https://github.com/OFAI/partitura.git@develop
```

* [Basis Mixer](https://github.com/OFAI/basismixer): Install from GitHub (is not yet in PyPI) and use branch `develop`.
```
pip install git+https://github.com/OFAI/basismixer.git@develop
```

TODO: write requirements and restructure the code as a package!

## Usage

The main program is `render_performance`. This program allows for generating an expressive performance given a score in MusicXML (preferred) or MIDI format.

The simplest way of generating a performance is
```bash
./bin/render_performance --outfile out.mid score_filename.xml
```

For more information see 
```bash
./bin/render_performance -h
```
