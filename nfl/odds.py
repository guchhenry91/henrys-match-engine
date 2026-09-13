"""Book prices, de-vigged, and the edge against them.

WHY THIS EXISTS. Everything else on this board is calibrated against the player's
own history: "over 21.5 rushing yards" means a better day than his typical one. It
does NOT mean better than the price a bookmaker is offering, and until this file
has real prices in it the board cannot claim an edge over anyone. Calibration and
profit are different things and only one of them has been demonstrated.

DE-VIGGING IS THE WHOLE POINT. A book's quoted prices imply probabilities summing
to more than 1 -- that excess is its margin. Comparing a model probability against
a RAW implied probability would show an edge on almost nothing, because the vig
is working against you before the model says a word. Removing it proportionally
gives the book's actual opinion, which is the only fair thing to disagree with.

THIS PARSER IS UNVERIFIED AGAINST LIVE DATA. As of 2026-08-27 the API returns no
odds for any NFL fixture -- week 1 is 14 days out and both bet365 and the
all-books query come back empty. So the shape below follows the documented
envelope and MUST fail loudly rather than guess: anything it cannot recognise is
skipped and counted, never coerced into a number. Building confidently on an
unverified feed is what produced 177 wrongly-deleted players earlier today.
"""

# bet365 as asked; the rest are fallbacks in rough order of how sharp their NFL
# prices tend to be. Pinnacle is last-but-sharpest deliberately: it is the best
# estimate of a true price, so it is the most honest thing to be measured against
# when bet365 is not quoting.
PREFERRED_BOOKS = ("Bet365", "Pinnacle", "WilliamHill", "Betfair", "Unibet", "888Sport")

# PLAYER-PROP BET TYPE IDS, READ FROM THE API'S OWN CATALOGUE on 2026-08-30
# (scripts/probe_nfl_odds.py), not inferred. These are the four markets the board
# publishes. The duplicates are real -- the catalogue lists "Player Passing Yards"
# at both 210 and 336 -- so both are accepted rather than one being guessed at.
#
# NOTHING IS PRICED YET. The same probe asked for the week-1 opener's odds by game
# id, from bet365 and from every book, and got zero records both times. Note that
# the PREVIOUS conclusion of "no odds" was reached with a broken query: it filtered
# by a `date` parameter the endpoint does not have, and the API was answering "The
# Date field do not exist." So the absence is now established properly rather than
# inherited. Having the ids recorded means that when prices do appear, no one has
# to guess which markets to ask for.
PLAYER_PROP_BETS = {
    "passing_yards": (210, 336),
    "rushing_yards": (236, 328),
    "receiving_yards": (266,),
    "anytime_touchdown": (47,),      # the catalogue calls it "Anytime Goal Scorer"
}

# The market that settles a team-winner pick. NFL has no draw in the regular
# season outside a rare tie, so a two-way price de-vigs cleanly.
MONEYLINE_MARKETS = ("Home/Away", "Moneyline", "Match Winner", "3Way Result")

# Below this an "edge" is inside the noise of both the model and the price, and
# publishing it would dress rounding up as an opportunity.
MIN_EDGE = 0.03


def decimal_to_prob(odd) -> float | None:
    """Implied probability from a decimal price. None if it is not a price."""
    try:
        value = float(odd)
    except (TypeError, ValueError):
        return None
    if value <= 1.0:
        return None                 # 1.0 pays nothing back; below that is nonsense
    return 1.0 / value


def devig(probabilities: dict) -> dict:
    """Scale a book's implied probabilities back to sum to 1.

    Proportional (multiplicative) rather than additive: additive de-vigging
    removes the same absolute amount from a 90% favourite and a 10% dog, which
    overstates the dog's true price badly. Proportional keeps their ratio, which
    is what the book's own pricing preserves.
    """
    total = sum(v for v in probabilities.values() if v is not None)
    if total <= 0:
        return {}
    return {k: v / total for k, v in probabilities.items() if v is not None}


def pick_bookmaker(bookmakers: list) -> dict | None:
    """The most preferred book that actually quoted this game."""
    by_name = {}
    for book in bookmakers or []:
        name = str(book.get("name") or "")
        if name:
            by_name.setdefault(name, book)
    for wanted in PREFERRED_BOOKS:
        for name, book in by_name.items():
            if wanted.lower().replace(" ", "") == name.lower().replace(" ", ""):
                return book
    # Anything is better than nothing, but say which so the card is not silently
    # measured against a book nobody would use.
    return next(iter(by_name.values()), None)


def moneyline(book: dict, home: str, away: str) -> dict | None:
    """{'home': p, 'away': p, 'book': name, 'raw': {...}} de-vigged, or None.

    Returns None rather than a guess whenever the market is missing, the values
    are unrecognisable, or a price fails to parse. A silent zero here would read
    as "the book thinks this is impossible" and manufacture a huge false edge.
    """
    if not book:
        return None
    for bet in book.get("bets") or []:
        name = str(bet.get("name") or "")
        if not any(m.lower() == name.lower() for m in MONEYLINE_MARKETS):
            continue
        raw = {}
        for value in bet.get("values") or []:
            label = str(value.get("value") or "").strip().lower()
            prob = decimal_to_prob(value.get("odd"))
            if prob is None:
                continue
            if label in ("home", "1", home.lower()):
                raw["home"] = prob
            elif label in ("away", "2", away.lower()):
                raw["away"] = prob
        if "home" in raw and "away" in raw:
            fair = devig(raw)
            return {"home": round(fair["home"], 4), "away": round(fair["away"], 4),
                    "book": book.get("name"),
                    "raw_home": round(raw["home"], 4),
                    "raw_away": round(raw["away"], 4),
                    "overround": round(sum(raw.values()), 4)}
    return None


import re

# "Bucky Irving - Over 50.5" -- the shape bet365 sends through API-NFL, read from
# the live response on 2026-09-13 (scripts/probe_nfl_props.py), not assumed.
_PROP_VALUE = re.compile(r"^(.*?)\s*-\s*(Over|Under)\s+(\d+(?:\.\d+)?)\s*$", re.I)


def player_props(book: dict) -> dict:
    """{market: {book_player_name: quote}} for every prop the book prices.

    Yards markets need BOTH sides at the same line, so the pair can be de-vigged
    into the book's fair over probability; a one-sided quote is refused. Where a
    player carries several lines (alternates), the one nearest an even price is
    kept -- that is the book's main line. Anytime TD is one-sided ("Bucky Irving"
    at 2.10), so its probability is the RAW implied one, margin included, and is
    flagged as such rather than dressed up as fair.
    """
    out = {}
    if not book:
        return out
    ids = {bid: m for m, bids in PLAYER_PROP_BETS.items() for bid in bids}
    for bet in book.get("bets") or []:
        try:
            market = ids.get(int(bet.get("id")))
        except (TypeError, ValueError):
            market = None
        if not market:
            continue
        quotes = out.setdefault(market, {})
        if market == "anytime_touchdown":
            for value in bet.get("values") or []:
                name = str(value.get("value") or "").strip()
                prob = decimal_to_prob(value.get("odd"))
                if name and prob is not None and name not in quotes:
                    quotes[name] = {"_name": name, "raw_yes": round(prob, 4),
                                    "odd": float(value["odd"]), "book": book.get("name")}
            continue
        sides = {}
        for value in bet.get("values") or []:
            match = _PROP_VALUE.match(str(value.get("value") or ""))
            prob = decimal_to_prob(value.get("odd"))
            if not match or prob is None:
                continue
            name, side, line = match.group(1).strip(), match.group(2).lower(), float(match.group(3))
            sides.setdefault((name, line), {})[side] = (prob, float(value["odd"]))
        for (name, line), pair in sides.items():
            if "over" not in pair or "under" not in pair:
                continue
            fair = devig({"over": pair["over"][0], "under": pair["under"][0]})
            quote = {"_name": name, "line": line, "over": round(fair["over"], 4),
                     "under": round(fair["under"], 4), "odd_over": pair["over"][1],
                     "odd_under": pair["under"][1], "book": book.get("name")}
            held = quotes.get(name)
            if held is None or abs(quote["over"] - 0.5) < abs(held["over"] - 0.5):
                quotes[name] = quote
    return {m: q for m, q in out.items() if q}


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """A player's name in a form both feeds agree on.

    The book writes "Kenneth Walker III" and "Marvin Harrison Jr."; nflverse
    writes "Kenneth Walker" and "Marvin Harrison". Punctuation goes ("D.J." ->
    "dj"), suffixes go, case goes. Nothing else is loosened -- a fuzzy match
    settles a bet against a stranger's yards.
    """
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def match_player(quotes: dict, player_name: str):
    """The book's quote for this player within ONE game, or None.

    `quotes` is that game's quotes, keyed by the book's player name (or carrying
    it as `_name`). None when nothing matches AND when more than one does:
    ambiguity is refused, never resolved by picking one.
    """
    want = norm_name(player_name)
    if not want:
        return None
    hits = [q for key, q in (quotes or {}).items()
            if norm_name(q.get("_name", key) if isinstance(q, dict) else key) == want]
    return hits[0] if len(hits) == 1 else None


def edge(model_prob: float, book_prob: float) -> float:
    """How much more likely the model thinks this is than the fair price implies."""
    return round(float(model_prob) - float(book_prob), 4)


def value_verdict(model_prob: float, book_prob: float) -> tuple[str, float]:
    """(verdict, edge). 'value' only when the gap clears MIN_EDGE.

    Three outcomes on purpose. "No value" is a real and useful answer -- most of
    the time a calibrated model and a sharp price agree, and a board that finds an
    edge on every game is describing its own error, not the market's.
    """
    gap = edge(model_prob, book_prob)
    if gap >= MIN_EDGE:
        return "value", gap
    if gap <= -MIN_EDGE:
        return "book favours", gap
    return "no edge", gap
