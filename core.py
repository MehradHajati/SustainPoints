# core.py
from __future__ import annotations

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Gemini model for insights
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_tokens=None,
    timeout=60,
    max_retries=3,
)

# ============ SUSTAINABILITY PARAMETERS ============

CATEGORY_WEIGHTS = {
    "public_transport": 90,
    "education": 85,
    "healthcare": 80,
    "groceries": 70,
    "utilities": 60,
    "restaurants": 50,
    "entertainment": 50,
    "travel": 40,
    "online_shopping": 35,
    "shopping": 30,
    "rideshare": 25,
    "gas": 20,
    "other": 50,
}

CARBON_FACTORS = {
    "gas": 400,  # g CO2/$
    "rideshare": 350,
    "travel": 250,
    "shopping": 150,
    "online_shopping": 155,
    "restaurants": 100,
    "utilities": 80,
    "entertainment": 60,
    "groceries": 50,
    "education": 30,
    "healthcare": 40,
    "public_transport": 40,
    "other": 100,
}

# ---------- DATA PREPARATION FUNCTIONS ----------


def extract_category_from_description(description: str) -> str:
    """Automatically infer a spending category from description text."""
    description = str(description).lower()
    category_map = {
        "groceries": [
            "grocery",
            "supermarket",
            "market",
            "whole foods",
            "trader joe",
            "costco",
            "loblaws",
        ],
        "restaurants": [
            "restaurant",
            "cafe",
            "coffee",
            "pizza",
            "burger",
            "dining",
            "ubereats",
            "doordash",
            "skiptheidishes",
        ],
        "gas": [
            "shell",
            "chevron",
            "bp",
            "exxon",
            "fuel",
            "petrol",
            "esso",
            "ultramar",
        ],
        "public_transport": [
            "metro",
            "bus",
            "train",
            "transit",
            "subway",
            "tram",
            "ticket",
            "ttc",
            "go transit",
        ],
        "rideshare": ["uber", "lyft", "taxi", "ride", "cab"],
        "shopping": [
            "mall",
            "retail",
            "store",
            "shopping",
            "walmart",
            "costco",
            "target",
        ],
        "online_shopping": ["amazon", "ebay", "etsy", "aliexpress", "shein"],
        "utilities": [
            "electric",
            "water",
            "internet",
            "bill",
            "phone",
            "utility",
            "hydro",
            "enbridge",
        ],
        "entertainment": [
            "movie",
            "cinema",
            "concert",
            "theater",
            "netflix",
            "spotify",
            "gaming",
        ],
        "healthcare": ["pharmacy", "doctor", "clinic", "hospital", "medical", "dental"],
        "education": ["school", "university", "tuition", "course", "training", "udemy"],
        "travel": [
            "hotel",
            "flight",
            "airline",
            "airbnb",
            "booking",
            "travel",
            "marriott",
        ],
    }

    for category, keywords in category_map.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "other"


def get_season(date_str: str) -> str:
    """Determine season (Northern Hemisphere)."""
    try:
        date_obj = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        month = date_obj.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    except Exception:
        return "Unknown"


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset by parsing dates, inferring categories, and adding metadata."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["category"] = df["description"].apply(extract_category_from_description)
    df["season"] = df["date"].apply(get_season)
    df = df[df["debit"] > 0]
    df["month"] = df["date"].dt.to_period("M")
    return df.sort_values("date")


def separate_by_season(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split dataframe into four seasonal subsets."""
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return {s: df[df["season"] == s].copy() for s in seasons}


# ============ SUSTAINABILITY SCORING FUNCTIONS ============


def calculate_sustainability_score(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Calculate weighted sustainability score (0-100).
    Formula: sum(amount * weight) / total_amount
    """
    if "debit" not in df.columns or "category" not in df.columns:
        return 0, {}

    total_spend = df["debit"].sum()
    if total_spend == 0:
        return 0, {}

    df["weight"] = df["category"].map(CATEGORY_WEIGHTS).fillna(50)
    score = (df["debit"] * df["weight"]).sum() / total_spend
    category_sums = df.groupby("category")["debit"].sum().to_dict()

    return min(score, 100), category_sums


def calculate_carbon_footprint(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate total CO2 equivalent and carbon intensity per dollar.
    Returns (total_kg_co2, kg_co2_per_dollar)
    """
    df = df.copy()
    df["carbon_factor"] = df["category"].map(CARBON_FACTORS).fillna(100)
    total_carbon_g = (df["debit"] * df["carbon_factor"]).sum()
    total_carbon_kg = total_carbon_g / 1000
    total_spend = df["debit"].sum()
    carbon_per_dollar = total_carbon_kg / total_spend if total_spend > 0 else 0

    return total_carbon_kg, carbon_per_dollar


def get_category_sustainability(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Breakdown sustainability metrics by category."""
    result = {}
    total_spend = df["debit"].sum()

    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        spend = cat_df["debit"].sum()
        weight = CATEGORY_WEIGHTS.get(category, 50)
        carbon_factor = CARBON_FACTORS.get(category, 100)
        carbon_kg = (spend * carbon_factor) / 1000

        result[category] = {
            "amount": spend,
            "percentage": (spend / total_spend * 100) if total_spend > 0 else 0,
            "weight": weight,
            "carbon_kg": carbon_kg,
            "transactions": len(cat_df),
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["amount"], reverse=True))


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Track sustainability score and spending by month."""
    monthly_data = []

    for month, month_df in df.groupby("month"):
        score, _ = calculate_sustainability_score(month_df)
        total_spend = month_df["debit"].sum()
        total_carbon, _ = calculate_carbon_footprint(month_df)

        monthly_data.append(
            {
                "month": str(month),
                "score": round(score, 2),
                "spending": total_spend,
                "carbon_kg": total_carbon,
                "transactions": len(month_df),
            }
        )

    return pd.DataFrame(monthly_data)


def get_seasonal_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detailed analysis per season."""
    result = {}
    seasonal_dfs = separate_by_season(df)

    for season, season_df in seasonal_dfs.items():
        if len(season_df) == 0:
            continue

        score, _ = calculate_sustainability_score(season_df)
        total_carbon, carbon_per_dollar = calculate_carbon_footprint(season_df)

        result[season] = {
            "score": round(score, 2),
            "spending": season_df["debit"].sum(),
            "carbon_kg": round(total_carbon, 2),
            "carbon_per_dollar": round(carbon_per_dollar, 4),
            "transactions": len(season_df),
            "avg_transaction": round(season_df["debit"].mean(), 2),
        }

    return result


def get_improvement_opportunities(
    df: pd.DataFrame, top_n: int = 5
) -> List[Dict[str, Any]]:
    """Identify top categories for sustainability improvement."""
    category_data = get_category_sustainability(df)

    opportunities = []
    for cat, data in category_data.items():
        weight = CATEGORY_WEIGHTS.get(cat, 50)
        if weight < 60:  # Low sustainability categories
            opportunities.append(
                {
                    "category": cat,
                    "current_spend": data["amount"],
                    "current_percentage": data["percentage"],
                    "sustainability_weight": weight,
                    "carbon_kg": data["carbon_kg"],
                    "improvement_potential": 100 - weight,
                }
            )

    return sorted(opportunities, key=lambda x: x["current_spend"], reverse=True)[:top_n]


# ============ VISUALIZATION DATA ============


def generate_pie_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Pie chart: spending by category."""
    category_data = df.groupby("category")["debit"].sum().sort_values(ascending=False)
    return {
        "labels": category_data.index.tolist(),
        "values": category_data.values.tolist(),
        "title": "Spending Distribution by Category",
    }


def generate_bar_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Bar chart: seasonal spending totals."""
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_data = df.groupby("season")["debit"].sum()
    season_data = season_data.reindex(
        [s for s in season_order if s in season_data.index]
    )

    return {
        "x": season_data.index.tolist(),
        "y": season_data.values.tolist(),
        "title": "Spending by Season",
    }


def generate_line_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Line chart: sustainability score trend over time."""
    monthly = get_monthly_trends(df)
    return {
        "x": monthly["month"].tolist(),
        "y": monthly["score"].tolist(),
        "title": "Sustainability Score Trend",
    }


def generate_carbon_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Bar chart: carbon footprint by category."""
    category_data = get_category_sustainability(df)
    categories = list(category_data.keys())
    carbon_values = [category_data[cat]["carbon_kg"] for cat in categories]

    return {
        "x": categories,
        "y": carbon_values,
        "title": "Carbon Footprint by Category (kg CO2)",
    }


# ============ LLM-POWERED INSIGHTS ============


def _build_sustainability_prompt() -> ChatPromptTemplate:
    """System prompt for comprehensive sustainability analysis."""
    return ChatPromptTemplate.from_template(
        """
You are a financial sustainability advisor analyzing credit card spending for environmental impact.

SPENDING ANALYSIS:
{summary_text}

CARBON FOOTPRINT:
{carbon_text}

IMPROVEMENT OPPORTUNITIES:
{opportunities_text}

Provide a comprehensive sustainability report with:

1. **Overall Assessment**: Rate their sustainability level (Excellent/Good/Fair/Needs Improvement)
2. **Key Findings**: Top 3 most impactful insights from their data
3. **Carbon Context**: Put their carbon footprint in context (trees needed to offset, etc.)
4. **Category Breakdown**: Which categories have the highest/lowest sustainability
5. **Seasonal Patterns**: Any notable seasonal trends
6. **Top 5 Recommendations**: Specific, actionable steps to improve sustainability
7. **Quick Wins**: Low-effort changes with high impact

Use markdown formatting. Keep under 400 words. Be encouraging but honest.

Your Response:
""".strip()
    )


def generate_sustainability_insights(analysis_result: Dict[str, Any]) -> str:
    """Generate comprehensive sustainability insights."""
    df = analysis_result["df"]
    score = analysis_result["score"]
    category_data = get_category_sustainability(df)
    seasonal_data = get_seasonal_analysis(df)
    opportunities = get_improvement_opportunities(df)
    carbon_total, carbon_per_dollar = calculate_carbon_footprint(df)

    # Build context texts
    summary_text = f"**Overall Score:** {score:.1f}/100\n"
    summary_text += f"**Total Spending:** ${df['debit'].sum():.2f}\n"
    summary_text += f"**Number of Transactions:** {len(df)}\n"
    summary_text += f"**Average Transaction:** ${df['debit'].mean():.2f}\n\n"
    summary_text += "**Top Spending Categories:**\n"
    for cat, data in list(category_data.items())[:5]:
        summary_text += (
            f"- {cat.title()}: ${data['amount']:.2f} ({data['percentage']:.1f}%)\n"
        )

    carbon_text = f"**Total CO2 Equivalent:** {carbon_total:.2f} kg\n"
    carbon_text += f"**Carbon Intensity:** {carbon_per_dollar:.4f} kg CO2/$\n"
    carbon_text += f"**Trees Needed to Offset:** ~{int(carbon_total / 20)}\n"
    carbon_text += "**Seasonal Breakdown:**\n"
    for season, data in seasonal_data.items():
        carbon_text += f"- {season}: {data['carbon_kg']:.2f} kg CO2\n"

    opportunities_text = "**Top Improvement Areas:**\n"
    for opp in opportunities:
        opportunities_text += f"- {opp['category'].title()}: {opp['improvement_potential']:.0f}% more sustainable alternative available\n"

    prompt = _build_sustainability_prompt()
    msgs = prompt.format_messages(
        summary_text=summary_text,
        carbon_text=carbon_text,
        opportunities_text=opportunities_text,
    )

    try:
        response = llm.invoke(msgs).content
    except Exception as e:
        response = f"Error generating insights: {str(e)}"

    return response


def _build_chat_prompt() -> ChatPromptTemplate:
    """Chat prompt for Q&A about sustainability."""
    return ChatPromptTemplate.from_template(
        """
You are a personal financial sustainability advisor. The user has uploaded their spending data.

CONTEXT:
Sustainability Score: {score}/100
Total CO2: {carbon_kg:.2f} kg
Monthly Average: ${monthly_avg:.2f}
Top Categories: {top_categories}

USER QUESTION: {question}

Provide helpful, data-specific answers about their spending sustainability. Use specific numbers from their data.
Keep responses under 150 words. Be encouraging and actionable.

Your Response:
""".strip()
    )


def answer_sustainability_question(
    question: str, analysis_result: Dict[str, Any]
) -> str:
    """Answer user questions about their sustainability."""
    df = analysis_result["df"]
    score = analysis_result["score"]
    carbon_total, _ = calculate_carbon_footprint(df)
    category_data = get_category_sustainability(df)

    top_cats = ", ".join(
        [
            f"{cat.title()} (${data['amount']:.0f})"
            for cat, data in list(category_data.items())[:3]
        ]
    )
    monthly_avg = df.groupby("month")["debit"].sum().mean()

    prompt = _build_chat_prompt()
    msgs = prompt.format_messages(
        score=f"{score:.1f}",
        carbon_kg=carbon_total,
        monthly_avg=monthly_avg,
        top_categories=top_cats,
        question=question,
    )

    try:
        response = llm.invoke(msgs).content
    except Exception as e:
        response = f"I encountered an error: {str(e)}"

    return response


# ============ MAIN PIPELINE ============


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Complete analysis pipeline."""
    df = preprocess_dataset(df)

    score, _ = calculate_sustainability_score(df)
    carbon_total, carbon_per_dollar = calculate_carbon_footprint(df)
    category_data = get_category_sustainability(df)
    seasonal_data = get_seasonal_analysis(df)
    monthly_trends = get_monthly_trends(df)

    result = {
        "df": df,
        "score": round(score, 2),
        "carbon_total": round(carbon_total, 2),
        "carbon_per_dollar": round(carbon_per_dollar, 4),
        "category_data": category_data,
        "seasonal_data": seasonal_data,
        "monthly_trends": monthly_trends,
        "total_spending": df["debit"].sum(),
        "transaction_count": len(df),
        "pie_data": generate_pie_data(df),
        "bar_data": generate_bar_data(df),
        "line_data": generate_line_data(df),
        "carbon_data": generate_carbon_data(df),
        "opportunities": get_improvement_opportunities(df),
    }

    result["insights"] = generate_sustainability_insights(result)

    return result
