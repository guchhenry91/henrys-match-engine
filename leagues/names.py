"""Canonical team names. Every source (football-data.co.uk, fixturedownload,
Understat, FBref, ClubElo) spells clubs differently — normalize once, here."""
import unicodedata


class UnknownTeam(Exception):
    """Raised when a source name has no canonical mapping — never guess."""


# canonical -> set of aliases seen across sources
ALIASES = {
    "PL": {
        "Manchester United": {"Man United", "Man Utd", "Manchester Utd"},
        "Manchester City": {"Man City"},
        "Newcastle United": {"Newcastle", "Newcastle Utd"},
        "Tottenham": {"Tottenham Hotspur", "Spurs"},
        "Wolves": {"Wolverhampton Wanderers", "Wolverhampton"},
        "Nottingham Forest": {"Nott'm Forest", "Nottingham", "Forest"},
        "Brighton": {"Brighton & Hove Albion", "Brighton and Hove Albion"},
        "West Ham": {"West Ham United"},
        "Leeds": {"Leeds United"},
        "Sunderland": set(),
        "Coventry": {"Coventry City"},
        "Arsenal": set(), "Chelsea": set(), "Liverpool": set(), "Everton": set(),
        "Aston Villa": set(), "Fulham": set(), "Brentford": set(),
        "Crystal Palace": set(), "Bournemouth": {"AFC Bournemouth"},
        "Hull": {"Hull City"},
        "Ipswich": {"Ipswich Town"},
        "Burnley": set(),
        "Leicester": {"Leicester City"},
        "Luton": {"Luton Town"},
        "Norwich": {"Norwich City"},
        "Sheffield United": {"Sheffield Utd"},
        "Southampton": set(),
        "Watford": set(),
        "West Brom": {"West Bromwich Albion", "West Bromwich"},
    },
    "LALIGA": {
        "Alaves": {"Deportivo Alavés", "Deportivo Alaves", "Alavés"},
        "Ath Bilbao": {"Athletic Club", "Athletic Bilbao", "Bilbao"},
        "Ath Madrid": {"Atlético Madrid", "Atletico Madrid", "Atlético de Madrid", "Atletico"},
        "Barcelona": {"FC Barcelona"},
        "Real Madrid": {"Real Madrid CF"},
        "Sociedad": {"Real Sociedad"},
        "Betis": {"Real Betis"},
        "Celta": {"Celta Vigo", "RC Celta"},
        "Getafe": {"Getafe CF"},
        "Sevilla": {"Sevilla FC"},
        "Valencia": {"Valencia CF"},
        "Villarreal": {"Villarreal CF"},
        "Espanol": {"Espanyol", "RCD Espanyol", "RCD Espanyol de Barcelona"},
        "Osasuna": {"CA Osasuna"},
        "Elche": {"Elche CF"},
        "Levante": {"Levante UD"},
        "Malaga": {"Málaga CF", "Malaga CF"},
        "Racing Santander": {"R. Racing Club", "Racing de Santander", "Real Racing Club", "Santander"},
        # "Dep. A Coruna" is football-data.co.uk's odds-feed spelling. Without it
        # the club's fixtures resolve everywhere else and silently lose their
        # market line -- today's Depor v Elche published with no odds at all.
        "La Coruna": {"RC Deportivo", "Deportivo La Coruna", "Deportivo", "Depor",
                      "Dep. A Coruna", "Dep. La Coruna", "Deportivo A Coruna"},
        "Vallecano": {"Rayo Vallecano"},
        "Almeria": {"UD Almeria", "UD Almería"},
        "Cadiz": {"Cadiz CF", "Cádiz CF", "Cádiz"},
        "Girona": {"Girona FC"},
        "Granada": {"Granada CF"},
        "Las Palmas": {"UD Las Palmas"},
        "Leganes": {"CD Leganes", "CD Leganés", "Leganés"},
        "Mallorca": {"RCD Mallorca"},
        "Oviedo": {"Real Oviedo"},
        "Valladolid": {"Real Valladolid"},
        "Eibar": {"SD Eibar"},
        "Huesca": {"SD Huesca"},
    },
    "BUNDESLIGA": {
        "Bayern Munich": {"FC Bayern München", "Bayern München", "Bayern Munchen", "Bayern"},
        "Dortmund": {"Borussia Dortmund", "BVB"},
        "Leverkusen": {"Bayer 04 Leverkusen", "Bayer Leverkusen"},
        "M'gladbach": {"Borussia Mönchengladbach", "Borussia Monchengladbach", "Borussia M.Gladbach", "Gladbach"},
        "Ein Frankfurt": {"Eintracht Frankfurt", "Frankfurt"},
        "Stuttgart": {"VfB Stuttgart"},
        "Wolfsburg": {"VfL Wolfsburg"},
        "RB Leipzig": {"RasenBallsport Leipzig"},
        "Union Berlin": {"1. FC Union Berlin"},
        "Werder Bremen": {"SV Werder Bremen", "Werder"},
        "Hoffenheim": {"TSG 1899 Hoffenheim", "TSG Hoffenheim", "1899 Hoffenheim"},
        "Freiburg": {"SC Freiburg", "Sport-Club Freiburg"},
        "Mainz": {"1. FSV Mainz 05", "Mainz 05", "FSV Mainz 05"},
        "Augsburg": {"FC Augsburg"},
        "Heidenheim": {"1. FC Heidenheim", "FC Heidenheim"},
        "St Pauli": {"FC St. Pauli", "St. Pauli"},
        "Hamburg": {"Hamburger SV", "Hamburg SV"},
        "Elversberg": {"SV Elversberg"},
        "FC Koln": {"1. FC Köln", "1. FC Koln", "Koln", "Köln", "FC Cologne", "Koeln"},
        "Schalke 04": {"FC Schalke 04", "Schalke"},
        "Paderborn": {"SC Paderborn 07"},
        "Bielefeld": {"Arminia Bielefeld", "DSC Arminia Bielefeld"},
        "Bochum": {"VfL Bochum"},
        "Darmstadt": {"SV Darmstadt 98", "Darmstadt 98"},
        "Greuther Furth": {"SpVgg Greuther Fürth", "SpVgg Greuther Furth", "Greuther Fürth", "Greuther Fuerth"},
        "Hertha": {"Hertha Berlin", "Hertha BSC"},
        "Holstein Kiel": {"KSV Holstein Kiel"},
    },
    # SERIE A. Canonical spelling is football-data.co.uk's, as in every other
    # league here, because that is where results and closing odds come from.
    # Built by reading all 27 club names each feed actually uses across the five
    # fitting seasons plus the current one, NOT from memory -- only three differ
    # at all, and a single unmapped name raises UnknownTeam mid-fit.
    "SERIEA": {
        "Milan": {"AC Milan", "A.C. Milan", "Milan AC"},
        "Inter": {"Internazionale", "Inter Milan", "FC Internazionale",
                  "Internazionale Milano"},
        "Parma": {"Parma Calcio 1913", "Parma Calcio"},
        "Verona": {"Hellas Verona", "Hellas Verona FC"},
        "Roma": {"AS Roma"},
        "Lazio": {"SS Lazio"},
        "Napoli": {"SSC Napoli"},
        "Juventus": {"Juventus FC", "Juve"},
        "Atalanta": {"Atalanta BC"},
        "Fiorentina": {"ACF Fiorentina"},
        "Torino": {"Torino FC"},
        "Bologna": {"Bologna FC", "Bologna 1909"},
        "Udinese": {"Udinese Calcio"},
        "Genoa": {"Genoa CFC"},
        "Cagliari": {"Cagliari Calcio"},
        "Lecce": {"US Lecce"},
        "Sassuolo": {"US Sassuolo", "Sassuolo Calcio"},
        "Como": {"Como 1907"},
        "Monza": {"AC Monza"},
        "Venezia": {"Venezia FC"},
        "Cremonese": {"US Cremonese"},
        "Pisa": {"Pisa SC", "AC Pisa"},
        "Frosinone": {"Frosinone Calcio"},
        "Empoli": {"Empoli FC"},
        "Salernitana": {"US Salernitana", "Salernitana 1919"},
        "Sampdoria": {"UC Sampdoria"},
        "Spezia": {"Spezia Calcio"},
    },
    "LIGUE1": {
        "Marseille": {"Olympique de Marseille", "Olympique Marseille"},
        "Paris SG": {"Paris Saint-Germain", "PSG", "Paris Saint Germain"},
        "Lyon": {"Olympique Lyonnais"},
        "Monaco": {"AS Monaco"},
        "Lille": {"LOSC Lille", "LOSC"},
        "Nice": {"OGC Nice"},
        "Rennes": {"Stade Rennais", "Stade Rennais FC"},
        "Lens": {"RC Lens"},
        "Strasbourg": {"RC Strasbourg Alsace", "RC Strasbourg"},
        "Nantes": {"FC Nantes"},
        "Toulouse": {"Toulouse FC"},
        "Brest": {"Stade Brestois", "Stade Brestois 29"},
        "Auxerre": {"AJ Auxerre"},
        "Angers": {"Angers SCO"},
        "Le Havre": {"Le Havre AC", "Havre Athletic Club"},
        "Metz": {"FC Metz"},
        "Lorient": {"FC Lorient"},
        "Paris FC": {"Paris FC"},
        "Troyes": {"Estac Troyes", "ESTAC Troyes"},
        "Le Mans": {"Le Mans FC"},
        "Ajaccio": {"AC Ajaccio"},
        "Bordeaux": {"FC Girondins de Bordeaux", "Girondins Bordeaux"},
        "Clermont": {"Clermont Foot"},
        "Montpellier": {"Montpellier HSC"},
        "Reims": {"Stade de Reims"},
        "St Etienne": {"Saint-Etienne", "AS Saint-Étienne", "Saint-Étienne", "ASSE"},
        "Dijon": {"Dijon FCO"},
        "Nimes": {"Nimes Olympique", "Nîmes", "Nîmes Olympique"},
    },
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _key(s: str) -> str:
    return _strip_accents(s).strip().lower()


# build reverse lookup once: normalized alias -> canonical
_LOOKUP: dict[str, dict[str, str]] = {}
for _lg, _mapping in ALIASES.items():
    table = {}
    for _canon, _aliases in _mapping.items():
        table[_key(_canon)] = _canon
        for _a in _aliases:
            table[_key(_a)] = _canon
    _LOOKUP[_lg] = table


def canonical(name: str, league: str) -> str:
    """Map any source spelling to our canonical club name."""
    table = _LOOKUP.get(league, {})
    hit = table.get(_key(name))
    if hit is None:
        raise UnknownTeam(f"{name!r} is not mapped for league {league!r}. "
                          f"Add it to leagues/names.py ALIASES.")
    return hit
