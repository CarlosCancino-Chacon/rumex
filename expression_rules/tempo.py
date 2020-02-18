import numpy as np

TEMPO_MARKINGS_BPM = {
    "larghissimo" : ( 18, 20),
    "grave": (20, 40),
    "largo": (40, 60),
    "larghetto": (60, 66),
    "adagio": (66, 76),
    "adagietto": (70, 80),
    "andante": (76, 108),
    "moderato": (108, 120),
    "allegretto": (112, 124),
    "allegro": (120, 168),
    "vivace": (140, 168),
    "vivacissimo": (168, 200),
    "presto": (168, 200),
    "prestissimo": (180, 210)}
    
# Larghissimo — very, very slow (20 bpm and below)
# Grave — slow and solemn (20–40 bpm)
# Lento — slowly (40–60 bpm)
# Largo — broadly (40–60 bpm)
# Larghetto — rather broadly (60–66 bpm)
# Adagio — slow and stately (literally, "at ease") (66–76 bpm)
# Adagietto — rather slow (70–80 bpm)
# Andante moderato — a bit slower than andante
# Andante — at a walking pace (76–108 bpm)
# Andantino – slightly faster than andante
# Moderato — moderately (108–120 bpm)
# Allegretto — moderately fast (but less so than allegro)
# Allegro moderato — moderately quick (112–124 bpm)
# Allegro — fast, quickly and bright (120–168 bpm)
# Vivace — lively and fast (≈140 bpm) (quicker than allegro)
# Vivacissimo — very fast and lively
# Allegrissimo — very fast
# Presto — very fast (168–200 bpm)
# Prestissimo — extremely fast (more than 200bpm)
def average_bp_by_tempo_marking(onsets, tempo_markings,
                                bp_per_tempo):

    beat_period = np.zeros_like(onsets)

    for tm in tempo_markings:
        print(tm)

    return beat_period






    
    
