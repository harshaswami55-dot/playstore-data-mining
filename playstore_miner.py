"""
=============================================================
PLAY STORE DATA MINER - Number Puzzle Games (India 50+)
Research Tool for Senior Gaming Habits Study
=============================================================
RUN ON YOUR LOCAL MACHINE:
    pip install google-play-scraper pandas

Optional sentiment model:
    pip install transformers torch

Then:
    python playstore_miner.py
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Callable

import pandas as pd
from google_play_scraper import Sort, app, reviews


# ---------------- CONFIG ----------------

TARGET_APPS = {
    "Number Match (Bitmango)": "com.bitmango.go.numbermatch",
    "Number Match Puzzle": "com.gram.games.numbermatch",
    "Sudoku": "com.easybrain.sudoku.android",
    "Woodoku": "com.tripledot.woodoku",
    "2048": "com.androbaby.game2048",
    "Brain Test": "com.unicostudio.braintest",
    "Ludo King": "com.ludo.king",
}

OBSERVATION_COLUMNS = [
    "user_id",
    "age",
    "gender",
    "urban_rural",
    "time_to_understand_rules",
    "wrong_taps",
    "asked_for_help",
    "font_issue",
    "language_issue",
    "ads_annoying",
    "overall_rating",
    "preferred_game",
]

INDIA_KEYWORDS_50PLUS = [
    "60 years",
    "65 years",
    "70 years",
    "retired",
    "senior citizen",
    "old age",
    "elderly",
    "grandfather",
    "grandmother",
    "dadi",
    "nana",
    "dada",
    "nani",
    "60 saal",
    "bujurg",
    "small font",
    "small text",
    "spectacles",
    "glasses",
    "arthritis",
    "arthriti",
    "shaky hands",
    "weak eyesight",
    "big button",
    "large text",
    "brain exercise",
    "mental exercise",
    "boring retirement",
    "morning chai",
    "kitty party",
]

PAIN_POINT_KEYWORDS = {
    "Font Too Small": ["small font", "small text", "can't read", "spectacles", "text size", "bigger font"],
    "Too Many Ads": ["too many ads", "ads come", "advertisement", "30 second ad", "ad popup", "ads"],
    "Difficulty Spike": ["too difficult", "too hard", "easy mode", "sudden hard", "impossible level"],
    "No Hindi/Language": ["hindi", "regional language", "english difficult", "mother tongue"],
    "Touch Issues": ["button small", "wrong tap", "arthritis", "shaky", "undo button", "fat fingers"],
    "Offline Needed": ["offline", "no internet", "2g", "network problem", "wifi needed"],
    "Privacy/Scam Fear": ["fraud", "scam", "permission", "bank detail", "suspicious", "fake"],
    "Crash/Technical": ["crash", "hang", "close", "restart", "not working", "login", "unresponsive"],
    "Monetization": ["coins", "pay to", "hint costs", "subscription", "money", "purchase"],
    "Tutorial Unclear": ["confusing", "no tutorial", "don't understand", "rules unclear"],
    "No Cloud Save": ["data deleted", "progress lost", "cloud save", "backup"],
    "Vision/Contrast": ["dark mode", "contrast", "color", "bright", "night mode"],
    "Dice/RNG Fairness": ["dice", "scripted", "algorithm", "fixed", "biased", "not random", "predictable"],
}


def clean_text(value) -> str:
    """Return lowercase text safely, even when Play Store fields are blank."""
    if pd.isna(value):
        return ""
    return str(value).lower()


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def build_sentiment_classifier() -> Callable[[str, int | None], str]:
    """
    Use Transformers when available. Fall back to a deterministic rating/keyword
    classifier so the script remains usable without large ML dependencies.
    """
    try:
        from transformers import pipeline

        sentiment_pipeline = pipeline("sentiment-analysis")

        def transformer_sentiment(text: str, score: int | None = None) -> str:
            if not text.strip():
                return "Neutral"
            result = sentiment_pipeline(text[:512])[0]
            label = result["label"].upper()
            confidence = float(result.get("score", 0))
            if confidence < 0.6:
                return "Neutral"
            if "NEGATIVE" in label:
                return "Negative"
            if "POSITIVE" in label:
                return "Positive"
            return "Neutral"

        print("  Sentiment: using Transformers pipeline")
        return transformer_sentiment
    except Exception as exc:
        print(f"  Sentiment: using lightweight fallback ({exc})")

    negative_words = [
        "bad",
        "worst",
        "fraud",
        "fake",
        "crash",
        "not working",
        "ads",
        "too many",
        "hang",
        "scripted",
        "biased",
    ]
    positive_words = ["good", "great", "best", "nice", "excellent", "love", "amazing", "super"]

    def fallback_sentiment(text: str, score: int | None = None) -> str:
        text = clean_text(text)
        if score is not None:
            if score <= 2:
                return "Negative"
            if score >= 4:
                return "Positive"
        if contains_any(text, negative_words):
            return "Negative"
        if contains_any(text, positive_words):
            return "Positive"
        return "Neutral"

    return fallback_sentiment


def scrape_app_metadata(app_id: str, app_name: str, country: str = "in") -> dict:
    """Fetch Play Store listing metadata for competitor comparison."""
    print(f"  Fetching metadata for {app_name}...")
    try:
        info = app(app_id, lang="en", country=country)
    except Exception as exc:
        print(f"    Metadata warning: {exc}")
        return {
            "app_name": app_name,
            "app_id": app_id,
            "title": app_name,
            "score": None,
            "ratings": None,
            "installs": None,
            "size": None,
            "updated": None,
            "contains_ads": None,
            "free": None,
            "genre": None,
        }

    return {
        "app_name": app_name,
        "app_id": app_id,
        "title": info.get("title"),
        "score": info.get("score"),
        "ratings": info.get("ratings"),
        "installs": info.get("installs"),
        "size": info.get("size"),
        "updated": info.get("updated"),
        "contains_ads": info.get("containsAds"),
        "free": info.get("free"),
        "genre": info.get("genre"),
    }


def scrape_reviews(
    app_id: str,
    app_name: str,
    count: int = 1000,
    hindi_count: int = 250,
    country: str = "in",
    delay_seconds: float = 1.5,
) -> pd.DataFrame:
    """Scrape English and Hindi Play Store reviews with error handling."""
    all_reviews = []
    print(f"\n  Scraping {app_name} ({app_id})...")

    for lang, lang_count in (("en", count), ("hi", min(hindi_count, count))):
        token = None
        scraped = 0
        batch_size = 100

        while scraped < lang_count:
            try:
                result, token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=min(batch_size, lang_count - scraped),
                    continuation_token=token,
                )
                for item in result:
                    item["review_language"] = lang
                all_reviews.extend(result)
                scraped += len(result)

                if not token or len(result) < batch_size:
                    break
                time.sleep(delay_seconds)
            except Exception as exc:
                print(f"    Review warning ({lang}): {exc}")
                break

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df["app_name"] = app_name
        df["app_id"] = app_id
        text = df["content"].fillna("").astype(str).str.lower()
        df["is_senior_likely"] = text.apply(lambda value: contains_any(value, INDIA_KEYWORDS_50PLUS))

    senior_count = int(df["is_senior_likely"].sum()) if not df.empty else 0
    print(f"  Got {len(df)} reviews ({senior_count} senior-likely)")
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    classifier = build_sentiment_classifier()
    df = df.copy()
    df["sentiment"] = df.apply(
        lambda row: classifier(row.get("content", ""), int(row["score"]) if not pd.isna(row["score"]) else None),
        axis=1,
    )
    return df


def analyze_pain_points(df: pd.DataFrame) -> dict:
    """Score each pain point by mentions and upvotes."""
    results = {}
    text = df["content"].fillna("").astype(str).str.lower()
    for pain, keywords in PAIN_POINT_KEYWORDS.items():
        mask = text.apply(lambda value: contains_any(value, keywords))
        subset = df[mask]
        results[pain] = {
            "count": int(mask.sum()),
            "upvotes": int(subset["thumbsUpCount"].fillna(0).sum()) if not subset.empty else 0,
            "avg_rating": float(subset["score"].mean()) if not subset.empty else 0,
            "negative_reviews": int((subset.get("sentiment") == "Negative").sum()) if "sentiment" in subset else 0,
            "sample_reviews": subset["content"].head(3).tolist(),
        }
    return dict(sorted(results.items(), key=lambda item: (item[1]["upvotes"], item[1]["count"]), reverse=True))


def build_competitor_comparison(reviews_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Create a senior-focused competitor comparison from metadata and review signals."""
    rows = []
    for app_name, group in reviews_df.groupby("app_name"):
        text = group["content"].fillna("").astype(str).str.lower()
        total = max(len(group), 1)
        senior_mentions = int(group["is_senior_likely"].sum())
        ads_mentions = int(text.apply(lambda value: contains_any(value, PAIN_POINT_KEYWORDS["Too Many Ads"])).sum())
        font_mentions = int(text.apply(lambda value: contains_any(value, PAIN_POINT_KEYWORDS["Font Too Small"])).sum())
        hindi_mentions = int(text.apply(lambda value: contains_any(value, PAIN_POINT_KEYWORDS["No Hindi/Language"])).sum())
        offline_mentions = int(text.apply(lambda value: contains_any(value, PAIN_POINT_KEYWORDS["Offline Needed"])).sum())
        difficulty_mentions = int(text.apply(lambda value: contains_any(value, PAIN_POINT_KEYWORDS["Difficulty Spike"])).sum())

        senior_score = 100
        senior_score -= min(30, ads_mentions * 100 / total)
        senior_score -= min(20, font_mentions * 100 / total * 2)
        senior_score -= min(15, difficulty_mentions * 100 / total * 2)
        senior_score += min(10, senior_mentions * 100 / total)
        senior_score = round(max(0, min(100, senior_score)), 1)

        rows.append(
            {
                "app_name": app_name,
                "reviews_analyzed": len(group),
                "review_avg_rating": round(float(group["score"].mean()), 2),
                "negative_review_pct": round(float((group["sentiment"] == "Negative").mean() * 100), 1),
                "ads_mentions": ads_mentions,
                "font_issue_mentions": font_mentions,
                "hindi_mentions": hindi_mentions,
                "offline_mentions": offline_mentions,
                "difficulty_mentions": difficulty_mentions,
                "senior_friendliness_score": senior_score,
            }
        )

    comparison = pd.DataFrame(rows)
    if metadata_df.empty:
        return comparison

    return comparison.merge(metadata_df, on="app_name", how="left")


def create_observation_template(path: str = "behavioral_observations_template.csv") -> None:
    if os.path.exists(path):
        return
    pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Observation template saved: {path}")


def analyze_behavioral_observations(path: str = "behavioral_observations.csv") -> dict:
    """Analyze live-demo observation data when the CSV exists."""
    if not os.path.exists(path):
        return {}

    obs = pd.read_csv(path)
    if obs.empty:
        return {}

    missing_columns = [column for column in OBSERVATION_COLUMNS if column not in obs.columns]
    if missing_columns:
        print(f"  Observation warning: missing columns {missing_columns}")
        return {}

    for column in ["age", "time_to_understand_rules", "wrong_taps", "overall_rating"]:
        if column in obs:
            obs[column] = pd.to_numeric(obs[column], errors="coerce")

    obs["age_group"] = pd.cut(
        obs["age"],
        bins=[0, 49, 59, 69, 120],
        labels=["Under 50", "50-59", "60-69", "70+"],
        include_lowest=True,
    )

    return {
        "rows": int(len(obs)),
        "avg_wrong_taps_by_age_group": obs.groupby("age_group", observed=True)["wrong_taps"].mean().round(2).dropna().to_dict(),
        "avg_time_by_urban_rural": obs.groupby("urban_rural")["time_to_understand_rules"].mean().round(2).dropna().to_dict(),
        "font_issue_rate_pct": round(float(obs["font_issue"].astype(str).str.lower().isin(["yes", "true", "1"]).mean() * 100), 1),
        "language_issue_rate_pct": round(float(obs["language_issue"].astype(str).str.lower().isin(["yes", "true", "1"]).mean() * 100), 1),
        "ads_annoying_rate_pct": round(float(obs["ads_annoying"].astype(str).str.lower().isin(["yes", "true", "1"]).mean() * 100), 1),
        "preferred_game_counts": obs["preferred_game"].value_counts().to_dict(),
    }


def generate_personas() -> list:
    """Research personas used to structure recommendations."""
    return [
        {
            "name": "Rameshji - The Retired Urban Professional",
            "age_range": "58-68",
            "location": "Tier-1 city (Pune/Jaipur/Lucknow)",
            "tech_comfort": "Medium - uses WhatsApp, YouTube",
            "play_time": "Morning and post-lunch, 30-60 min/day",
            "motivation": "Mental exercise, combating boredom after retirement",
            "frustrations": ["Small font size", "Sudden difficulty spikes", "Intrusive ads"],
            "wishlist": ["Bigger buttons", "Senior mode", "Dark mode"],
            "quote": "I play after my morning walk. This keeps my brain sharp but font is very small.",
        },
        {
            "name": "Savitribai - The First-Time Rural Gamer",
            "age_range": "55-70",
            "location": "Tier-3 town or village (Bihar/Rajasthan/UP)",
            "tech_comfort": "Low - relies on family for installation",
            "play_time": "Afternoon free time, irregular",
            "motivation": "Timepass, curiosity after family showed the game",
            "frustrations": ["No Hindi UI", "Needs internet", "Storage limit on phone"],
            "wishlist": ["Full Hindi support", "Offline mode", "Smaller app size"],
            "quote": "Meri bahu ne install kiya. Hindi mein hota toh main khud khelti.",
        },
        {
            "name": "Nalini Amma - The Social Senior",
            "age_range": "60-75",
            "location": "South Indian metro/city (Chennai/Hyderabad/Bangalore)",
            "tech_comfort": "Medium-Low - plays with help of children/grandchildren",
            "play_time": "Evening with friends/WhatsApp group, 1-2 hrs/day",
            "motivation": "Social bonding, competition with friends, routine",
            "frustrations": ["Touch accuracy", "Scary ads/permissions", "No progress save"],
            "wishlist": ["Larger touch targets", "Ad-free experience", "Leaderboard for seniors"],
            "quote": "Our kitty group of 8 ladies all play. We compare scores. But ads are too irritating.",
        },
    ]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No data available._\n"

    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return "_No matching columns available._\n"

    table = df[available_columns].fillna("")
    lines = [
        "| " + " | ".join(available_columns) + " |",
        "| " + " | ".join(["---"] * len(available_columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[column]).replace("|", "/") for column in available_columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    competitor_df: pd.DataFrame,
    pain_points: dict,
    personas: list,
    observations: dict,
    output_path: str,
) -> None:
    senior_df = df[df["is_senior_likely"]].copy()
    analysis_df = senior_df if not senior_df.empty else df
    frustrated_ads = 0
    if not analysis_df.empty:
        ads_mask = analysis_df["content"].fillna("").astype(str).str.lower().apply(
            lambda value: contains_any(value, PAIN_POINT_KEYWORDS["Too Many Ads"])
        )
        frustrated_ads = round(float((ads_mask & (analysis_df["sentiment"] == "Negative")).mean() * 100), 1)

    report = f"""# Play Store Mining Report - Number Puzzle Games
## Indian Users Aged 50+ | Research Assignment

**Generated:** {datetime.now().strftime('%B %d, %Y')}
**Total Reviews Analyzed:** {len(df)}
**Average Rating:** {df['score'].mean():.2f} / 5
**Senior-Likely Reviews:** {int(df['is_senior_likely'].sum())}

---

## 1. Key Statistics

| Metric | Value |
|--------|-------|
| Total Reviews | {len(df)} |
| Average Rating | {df['score'].mean():.2f} stars |
| Positive Reviews | {(df['sentiment'] == 'Positive').sum()} |
| Neutral Reviews | {(df['sentiment'] == 'Neutral').sum()} |
| Negative Reviews | {(df['sentiment'] == 'Negative').sum()} |
| Senior-Likely Reviews | {int(df['is_senior_likely'].sum())} ({df['is_senior_likely'].mean()*100:.1f}%) |
| Senior/Analysis Users Frustrated With Ads | {frustrated_ads}% |

---

## 2. App Metadata

{markdown_table(metadata_df, ['app_name', 'score', 'ratings', 'installs', 'size', 'updated'])}

---

## 3. Competitor Comparison For Seniors

{markdown_table(competitor_df, ['app_name', 'reviews_analyzed', 'review_avg_rating', 'negative_review_pct', 'ads_mentions', 'font_issue_mentions', 'hindi_mentions', 'offline_mentions', 'difficulty_mentions', 'senior_friendliness_score'])}

---

## 4. Pain Points Ranked By Community Impact

| # | Pain Point | Mentions | Total Upvotes | Avg Rating | Negative Reviews |
|---|------------|----------|---------------|------------|------------------|
"""
    for i, (pain, stats) in enumerate(pain_points.items(), 1):
        report += (
            f"| {i} | {pain} | {stats['count']} | {stats['upvotes']} | "
            f"{stats['avg_rating']:.1f} stars | {stats['negative_reviews']} |\n"
        )

    report += "\n---\n\n## 5. User Personas\n\n"
    for persona in personas:
        report += f"""### {persona['name']}
- **Age:** {persona['age_range']} | **Location:** {persona['location']}
- **Tech Comfort:** {persona['tech_comfort']}
- **Play Pattern:** {persona['play_time']}
- **Motivation:** {persona['motivation']}
- **Top Frustrations:** {', '.join(persona['frustrations'])}
- **Feature Wishlist:** {', '.join(persona['wishlist'])}
- *"{persona['quote']}"*

"""

    report += """---

## 6. Feature Recommendations

1. **Larger font and touch targets** - prioritize readability and reduce wrong taps.
2. **Senior/easy mode** - slower difficulty curve, extra hints, and clearer onboarding.
3. **Reduced or skippable ads** - especially during active gameplay.
4. **Hindi and regional-language UI** - important for rural and first-time smartphone users.
5. **Offline mode** - supports users with weak or expensive data connections.
6. **High-contrast/dark mode** - reduces eye strain during evening use.
7. **Progress backup** - prevents frustration when phones are changed or apps are reinstalled.

---

## 7. Behavioral Observation Analysis

"""
    if observations:
        report += f"""| Metric | Result |
|--------|--------|
| Participants | {observations['rows']} |
| Font Issue Rate | {observations['font_issue_rate_pct']}% |
| Language Issue Rate | {observations['language_issue_rate_pct']}% |
| Ads Annoying Rate | {observations['ads_annoying_rate_pct']}% |

**Average wrong taps by age group:** {observations['avg_wrong_taps_by_age_group']}

**Average rule-understanding time by urban/rural:** {observations['avg_time_by_urban_rural']}

**Preferred game counts:** {observations['preferred_game_counts']}
"""
    else:
        report += (
            "No `behavioral_observations.csv` file found yet. "
            "Use `behavioral_observations_template.csv` during the live demo, then rerun this script.\n"
        )

    report += "\n---\n*Report generated by Play Store Data Mining Pipeline*\n"

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"  Report saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Google Play reviews for senior-focused puzzle game research.")
    parser.add_argument(
        "--reviews-per-app",
        type=int,
        default=1000,
        help="Maximum English reviews to scrape per app. Default: 1000.",
    )
    parser.add_argument(
        "--hindi-reviews-per-app",
        type=int,
        default=250,
        help="Maximum Hindi reviews to scrape per app. Default: 250.",
    )
    parser.add_argument(
        "--country",
        default="in",
        help="Google Play country code. Default: in.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.5,
        help="Pause between review batches to avoid scraper throttling. Default: 1.5.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for generated output files, for example large_ creates large_all_reviews.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix = args.output_prefix
    print("=" * 60)
    print("PLAY STORE DATA MINER - Starting...")
    print(f"Reviews per app: {args.reviews_per_app} English + {args.hindi_reviews_per_app} Hindi")
    print(f"Country: {args.country}")
    print(f"Output prefix: {prefix or '(none)'}")
    print("=" * 60)

    metadata_rows = []
    review_dfs = []

    for app_name, app_id in TARGET_APPS.items():
        metadata_rows.append(scrape_app_metadata(app_id, app_name, country=args.country))
        df = scrape_reviews(
            app_id,
            app_name,
            count=args.reviews_per_app,
            hindi_count=args.hindi_reviews_per_app,
            country=args.country,
            delay_seconds=args.delay_seconds,
        )
        if not df.empty:
            review_dfs.append(df)
        time.sleep(2)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = f"{prefix}app_metadata.csv"
    reviews_path = f"{prefix}all_reviews.csv"
    competitor_path = f"{prefix}competitor_comparison.csv"
    report_path = f"{prefix}playstore_report.md"

    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"  Metadata saved: {metadata_path}")

    if not review_dfs:
        print("No data scraped. Check internet connection and app IDs.")
        return 1

    combined = pd.concat(review_dfs, ignore_index=True)
    combined.drop_duplicates(subset="reviewId", inplace=True)
    combined = add_sentiment(combined)
    combined.to_csv(reviews_path, index=False, encoding="utf-8-sig")
    print(f"\nTotal unique reviews: {len(combined)}")
    print(f"Senior-likely reviews: {int(combined['is_senior_likely'].sum())}")

    competitor_df = build_competitor_comparison(combined, metadata_df)
    competitor_df.to_csv(competitor_path, index=False, encoding="utf-8-sig")
    print(f"  Competitor comparison saved: {competitor_path}")

    create_observation_template()
    observations = analyze_behavioral_observations()
    pain_points = analyze_pain_points(combined)
    personas = generate_personas()

    generate_report(
        combined,
        metadata_df,
        competitor_df,
        pain_points,
        personas,
        observations,
        report_path,
    )
    print(f"\nDone! Check: {reviews_path}, {metadata_path}, {competitor_path}, and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
