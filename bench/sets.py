"""The bench's two sources of other matches, behind one interface.

    idsse     7 DFL matches, labels from DFL's per-frame fields (bench/idsse.py, bench/labels.py)
    metrica   3 Metrica sample games, labels from Metrica's events (bench/metrica.py)
"""

from bench import idsse, labels, metrica

SETS = ("idsse", "metrica", "all")


def matches(which):
    return {"idsse": idsse.matches(), "metrica": metrica.match_names(),
            "all": idsse.matches() + metrica.match_names()}[which]


def source(match):
    return "metrica" if match.startswith("metrica") else "idsse"


def load(match):
    return metrica.load(match) if source(match) == "metrica" else idsse.load(match)


def truth(match, data):
    """Scorer codes for every frame: Metrica's from its events, IDSSE's from DFL's fields."""
    return data["labels"] if source(match) == "metrica" else labels.codes(data)[0]
