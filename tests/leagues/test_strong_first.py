"""Every board lists Strong (70%+) picks first, then Solid (60-70%), then the rest."""
import pathlib
import re

HTML = (pathlib.Path(__file__).resolve().parents[2] / "index.html").read_text(encoding="utf-8")


def test_the_tier_rank_puts_strong_then_solid_first():
    body = re.search(r"function tierRank\(x\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert "b>=5?0:b===4?1:2" in body


def test_every_board_uses_it():
    # Today, Best Picks, soccer props, NFL games + props, NBA games + props, UCL.
    assert HTML.count(".sort(strongFirst(") == 8
