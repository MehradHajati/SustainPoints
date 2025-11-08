# app.py
import os
from dotenv import load_dotenv
import uuid
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from core import run_full_analysis, answer_sustainability_question

load_dotenv()

# Session storage
_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


# ============ VISUALIZATION FUNCTIONS ============


def create_pie_chart(chart_data):
    """Sustainability-focused pie chart."""
    if not chart_data or "labels" not in chart_data:
        return None

    colors = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c", "#c0392b", "#95a5a6"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                marker=dict(colors=colors),
                textposition="inside",
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        height=450,
        title={"text": chart_data.get("title", ""), "font": {"size": 16}},
        font=dict(size=12),
    )
    return fig


def create_bar_chart(chart_data, title=""):
    """Seasonal bar chart."""
    if not chart_data or "x" not in chart_data:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["x"],
                y=chart_data["y"],
                marker=dict(
                    color=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"][
                        : len(chart_data["x"])
                    ]
                ),
                text=[f"${v:.0f}" for v in chart_data["y"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title={"text": chart_data.get("title", title), "font": {"size": 16}},
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        height=400,
        font=dict(size=12),
        showlegend=False,
    )
    return fig


def create_line_chart(chart_data):
    """Trend line chart."""
    if not chart_data or "x" not in chart_data:
        return None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=chart_data["x"],
                y=chart_data["y"],
                mode="lines+markers",
                fill="tozeroy",
                name="Sustainability Score",
                line=dict(color="#3498db", width=3),
                marker=dict(size=8, color="#2980b9"),
            )
        ]
    )
    fig.update_layout(
        title={"text": chart_data.get("title", "Score Trend"), "font": {"size": 16}},
        xaxis_title="Month",
        yaxis_title="Score (0-100)",
        height=400,
        font=dict(size=12),
        showlegend=False,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def create_score_gauge(score):
    """Sustainability score gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Sustainability Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "#2ecc71"
                    if score >= 70
                    else "#f39c12"
                    if score >= 50
                    else "#e74c3c"
                },
                "steps": [
                    {"range": [0, 33], "color": "#ffe6e6"},
                    {"range": [33, 66], "color": "#fff9e6"},
                    {"range": [66, 100], "color": "#e6ffe6"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )
    fig.update_layout(height=400, font=dict(size=12))
    return fig


# ============ MAIN ANALYSIS FUNCTION ============


def analyze_credit_data(file):
    """Process and analyze credit card data."""
    if file is None:
        return "📋 Please upload a CSV file", None, None, None, None, None, {}

    try:
        df = pd.read_csv(file.name)
        required_cols = {"date", "description", "credit", "debit"}
        if not required_cols.issubset(df.columns):
            return (
                f"❌ Missing columns: {required_cols - set(df.columns)}",
                None,
                None,
                None,
                None,
                None,
                {},
            )

        result = run_full_analysis(df)

        # Summary metrics
        score = result["score"]
        carbon = result["carbon_total"]
        spending = result["total_spending"]
        transactions = result["transaction_count"]

        # Report text
        report = f"""
# 🌿 Sustainability Report

## Overview
- **Sustainability Score:** `{score}/100`
- **Total CO₂ Equivalent:** `{carbon:.2f} kg` 🌍
- **Total Spending:** `${spending:,.2f}`
- **Transactions:** `{transactions}`

## Environmental Impact
- **Carbon per Dollar:** `{result["carbon_per_dollar"]:.4f} kg CO₂/$`
- **Trees Needed to Offset:** ~`{int(carbon / 20)}`

---

## Detailed Analysis

{result["insights"]}
"""

        return (
            report,
            create_score_gauge(score),
            create_pie_chart(result["pie_data"]),
            create_bar_chart(result["bar_data"]),
            create_line_chart(result["line_data"]),
            create_bar_chart(result["carbon_data"], "Carbon Footprint"),
            result,
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None, None, {}


# ============ CHATBOT FUNCTION ============


def chatbot_reply(message, history, analysis_state, session_state):
    """Chat with context from analysis."""
    if not analysis_state or "insights" not in analysis_state:
        return "Please analyze your spending data first! 📊", session_state

    session_id = session_state.get("session_id")
    if not session_id:
        session_id = f"session-{uuid.uuid4().hex}"
        session_state["session_id"] = session_id

    user_msg = message.get("text") if isinstance(message, dict) else message
    response = answer_sustainability_question(user_msg, analysis_state)

    return response, session_state


# ============ GRADIO UI (HALIFAX DESIGN STANDARDS) ============

with gr.Blocks(
    title="🌿 Sustainability Points Analyzer",
    theme=gr.themes.Soft(),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .header-title {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .upload-area {
        border: 2px dashed #2ecc71;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f0fff4;
    }
    
    .action-button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border: none;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
    }
    """,
) as demo:
    # Header
    with gr.Group(elem_classes="header-title"):
        gr.Markdown("""
# 🌿 Sustainability Points Analyzer
*Measure, Understand & Improve Your Spending Sustainability*
        """)

    # Session state
    analysis_state = gr.State({})
    session_state = gr.State({})

    # ========== TAB 1: ANALYSIS ==========
    with gr.Tab("📊 Analysis"):
        with gr.Group():
            gr.Markdown("### 📤 Upload Your Credit Card Data")
            gr.Markdown(
                "*CSV format: date, description, credit, debit (supports DD/MM/YYYY)*"
            )

            with gr.Row():
                file_upload = gr.File(
                    label="Choose CSV File",
                    file_types=[".csv"],
                    elem_classes="upload-area",
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

        # Score gauge
        with gr.Group():
            gr.Markdown("### Your Sustainability Score")
            score_gauge = gr.Plot(label="Sustainability Gauge")

        # Main report
        with gr.Group():
            gr.Markdown("### 📋 Detailed Report")
            report_text = gr.Markdown()

        # Visualizations
        gr.Markdown("### 📈 Visualizations")
        with gr.Row():
            pie_chart = gr.Plot(label="Spending Distribution")
            seasonal_chart = gr.Plot(label="Seasonal Breakdown")

        with gr.Row():
            trend_chart = gr.Plot(label="Sustainability Trend")
            carbon_chart = gr.Plot(label="Carbon Footprint")

        # Analysis button click
        analyze_btn.click(
            fn=analyze_credit_data,
            inputs=[file_upload],
            outputs=[
                report_text,
                score_gauge,
                pie_chart,
                seasonal_chart,
                trend_chart,
                carbon_chart,
                analysis_state,
            ],
        )

    # ========== TAB 2: CHATBOT ==========
    with gr.Tab("💬 Ask Me Anything"):
        gr.Markdown("""
### 🤖 Sustainability Assistant
Ask questions about your spending habits, get personalized recommendations, and learn how to improve!

**Example questions:**
- "How can I improve my sustainability?"
- "Which category should I focus on?"
- "How does my score compare?"
- "What are quick wins?"
        """)

        chatbot = gr.ChatInterface(
            fn=chatbot_reply,
            additional_inputs=[analysis_state, session_state],
            additional_outputs=[session_state],
            type="messages",
            fill_height=True,
        )

    # ========== TAB 3: GUIDE ==========
    with gr.Tab("📚 Sustainability Guide"):
        gr.Markdown("""
## Understanding Sustainability Scores

### ✅ What is a Good Score?
- **80-100:** Excellent! You're a sustainability champion 🏆
- **60-79:** Good! Keep pushing for improvement 👍
- **40-59:** Moderate - There's room to improve 📈
- **0-39:** Needs work - Let's make a plan 💪

### 🌍 Category Sustainability Weights
- **Highest (80+):** Public Transport, Education, Healthcare
- **High (60-79):** Groceries, Utilities
- **Medium (40-59):** Restaurants, Entertainment
- **Low (20-39):** Shopping, Rideshare, Gas

### 💡 Quick Wins
1. Replace 2 rideshare trips with public transit (+10 points)
2. Buy groceries locally (+5 points)
3. Cancel unused subscriptions (+3 points)
4. Use public transit for daily commute (+25 points)
5. Shop second-hand for clothing (+8 points)

### 🌱 Carbon Context
- 1 kg CO₂ = Tree planted equivalent
- Average person: 5-10 kg CO₂/day
- Target: < 2 kg CO₂/day

### 📊 How We Calculate
**Score = Weighted average of all spending**
- Each category has a sustainability weight (0-100)
- Score reflects your overall spending sustainability
- Monthly trends show your progress

### 🌳 Offset Your Carbon
- Plant trees through carbon offset programs
- Support renewable energy initiatives
- Invest in sustainable businesses
        """)

if __name__ == "__main__":
    demo.launch(share=False)
