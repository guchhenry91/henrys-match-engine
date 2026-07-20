"""Refresh imminent Best-Pick team news from free independent RSS indexes."""
from leagues.team_news import write_refreshed


def main() -> None:
    news = write_refreshed()
    checked = sum(
        1 for league, clubs in news.items()
        if not league.startswith("_") and isinstance(clubs, dict)
        for entry in clubs.values()
        if isinstance(entry, dict) and entry.get("checked")
    )
    print(f"team news refreshed; {checked} clubs have a complete automatic check")


if __name__ == "__main__":
    main()
