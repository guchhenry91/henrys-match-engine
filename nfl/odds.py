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
