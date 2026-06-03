"""
Phase 2: Ecommerce AI Agent + A/B Testing Extension
---------------------------------------------------
This version directly extends the existing Local Ecommerce Growth Agent.

Phase 1:
The agent diagnoses an ecommerce issue and recommends growth levers.

Phase 2:
The experiment engine converts those levers into a testable A/B experiment:
- Control: current/default ecommerce experience
- Treatment: agent-recommended optimization strategy
- KPI: conversion rate, CTR, revenue per user, or AOV
- Analysis: uplift, statistical significance, and segment-level impact

This makes the project story stronger:
"The AI agent does not just recommend a strategy. It creates a business hypothesis that can be validated through experimentation."
"""

from ollama import chat
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt


# -----------------------------
# Phase 1: Existing Ecommerce Agent Logic
# -----------------------------

def ecommerce_levers(site_type: str, product_type: str, customer_type: str, goal: str, issue: str, budget: str) -> list[str]:
    text = f"{site_type} {product_type} {customer_type} {goal} {issue} {budget}".lower()

    levers = []

    if "traffic" in text and ("low sales" in text or "low conversion" in text):
        levers.append("landing_page_relevance")
        levers.append("product_page_clarity")
        levers.append("checkout_friction")

    if "cart" in text or "abandon" in text:
        levers.append("cart_recovery")
        levers.append("shipping_transparency")
        levers.append("urgency_messaging")

    if "repeat" in text or "retention" in text:
        levers.append("lifecycle_email")
        levers.append("loyalty_offer")
        levers.append("customer_segmentation")

    if "new customer" in text or "acquisition" in text:
        levers.append("first_purchase_offer")
        levers.append("social_proof")
        levers.append("creator_content")

    if "average order" in text or "aov" in text:
        levers.append("bundles")
        levers.append("free_shipping_threshold")
        levers.append("cross_sell_recommendations")

    if "low" in text:
        levers.append("low_cost_optimization")
        levers.append("reviews")
        levers.append("retargeting")

    if not levers:
        levers = [
            "customer_journey_review",
            "product_page_optimization",
            "email_retargeting",
            "funnel_metric_tracking"
        ]

    return levers


def build_prompt(site_type: str, product_type: str, customer_type: str, goal: str, issue: str, budget: str) -> str:
    levers = ecommerce_levers(site_type, product_type, customer_type, goal, issue, budget)
    lever_text = "\n".join(f"- {lever.replace('_', ' ')}" for lever in levers)

    return f"""
You are an ecommerce growth strategist with strong analytics instincts.

Website context:
Website type: {site_type}
Product type: {product_type}
Customer type: {customer_type}
Business goal: {goal}
Current issue: {issue}
Budget: {budget}

Suggested ecommerce levers:
{lever_text}

Please provide:
1. A short diagnosis of what may be happening
2. The top ecommerce levers to focus on
3. Three campaign or optimization ideas
4. Product page, site, or checkout improvements
5. Recommended KPIs to track
6. A simple 7 day action plan

Make the answer useful for this specific ecommerce website.
Keep it practical, specific, and analytics-minded.
""".strip()


# -----------------------------
# Phase 2: Convert Agent Strategy into A/B Test Brief
# -----------------------------

def build_experiment_brief(site_type: str, product_type: str, customer_type: str, goal: str, issue: str, budget: str) -> dict:
    """Turn the Ecommerce AI Agent's recommended levers into a testable experiment."""

    levers = ecommerce_levers(site_type, product_type, customer_type, goal, issue, budget)
    text = " ".join(levers + [goal, issue, customer_type]).lower()

    if "cart_recovery" in levers or "abandon" in text:
        strategy_name = "AI-assisted cart recovery messaging"
        control = "Standard abandoned-cart email"
        treatment = "Personalized cart recovery message using urgency, shipping clarity, and product-specific reminders"
        primary_kpi = "conversion_rate"
        expected_lift = 0.12

    elif "cross_sell_recommendations" in levers or "bundles" in levers or "aov" in text:
        strategy_name = "AI-personalized cross-sell recommendations"
        control = "Generic product recommendations"
        treatment = "Personalized bundles and cross-sell recommendations based on customer intent"
        primary_kpi = "revenue_per_user"
        expected_lift = 0.10

    elif "first_purchase_offer" in levers or "acquisition" in text or "new customer" in text:
        strategy_name = "AI-personalized first-purchase offer"
        control = "Generic welcome offer"
        treatment = "Segment-aware first-purchase offer with personalized product messaging"
        primary_kpi = "conversion_rate"
        expected_lift = 0.14

    elif "lifecycle_email" in levers or "retention" in text or "repeat" in text:
        strategy_name = "AI-assisted lifecycle retention campaign"
        control = "Standard promotional email"
        treatment = "Lifecycle email personalized by segment, prior behavior, and replenishment timing"
        primary_kpi = "repeat_purchase_rate"
        expected_lift = 0.09

    else:
        strategy_name = "AI-assisted product page optimization"
        control = "Current product page experience"
        treatment = "Improved product page with clearer benefits, reviews, FAQs, and trust signals"
        primary_kpi = "conversion_rate"
        expected_lift = 0.08

    return {
        "strategy_name": strategy_name,
        "business_goal": goal,
        "current_issue": issue,
        "target_customer": customer_type,
        "control_experience": control,
        "treatment_experience": treatment,
        "primary_kpi": primary_kpi,
        "expected_lift": expected_lift,
        "agent_levers": levers,
        "hypothesis": f"If we replace the control experience with {treatment.lower()}, then {primary_kpi.replace('_', ' ')} will improve for {customer_type}."
    }


# -----------------------------
# Phase 2: Synthetic Experiment Simulation
# -----------------------------

def generate_experiment_data(experiment_brief: dict, n_users: int = 50_000) -> pd.DataFrame:
    """Simulate ecommerce user behavior based on the agent-generated experiment brief."""

    np.random.seed(42)

    segments = ["new", "loyal", "discount_shopper", "high_value"]
    traffic_sources = ["organic", "paid_search", "email", "social", "direct"]
    devices = ["mobile", "desktop", "tablet"]

    df = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "customer_segment": np.random.choice(segments, n_users, p=[0.40, 0.25, 0.25, 0.10]),
        "traffic_source": np.random.choice(traffic_sources, n_users, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
        "device_type": np.random.choice(devices, n_users, p=[0.60, 0.32, 0.08]),
        "experiment_group": np.random.choice(["control", "treatment"], n_users, p=[0.50, 0.50])
    })

    base_conversion = {
        "new": 0.025,
        "loyal": 0.055,
        "discount_shopper": 0.040,
        "high_value": 0.070
    }

    base_ctr = {
        "new": 0.055,
        "loyal": 0.085,
        "discount_shopper": 0.075,
        "high_value": 0.095
    }

    avg_order_value = {
        "new": 55,
        "loyal": 82,
        "discount_shopper": 48,
        "high_value": 135
    }

    df["base_conversion_rate"] = df["customer_segment"].map(base_conversion)
    df["base_ctr"] = df["customer_segment"].map(base_ctr)
    df["avg_order_value"] = df["customer_segment"].map(avg_order_value)

    # The treatment effect comes from the agent-generated experiment brief.
    expected_lift = experiment_brief["expected_lift"]

    # Add realistic variation by segment.
    segment_multiplier = {
        "new": 1.25,
        "loyal": 0.70,
        "discount_shopper": 1.05,
        "high_value": 0.90
    }

    df["segment_multiplier"] = df["customer_segment"].map(segment_multiplier)

    df["final_conversion_rate"] = np.where(
        df["experiment_group"] == "treatment",
        df["base_conversion_rate"] * (1 + expected_lift * df["segment_multiplier"]),
        df["base_conversion_rate"]
    )

    df["final_ctr"] = np.where(
        df["experiment_group"] == "treatment",
        df["base_ctr"] * (1 + expected_lift * 0.75 * df["segment_multiplier"]),
        df["base_ctr"]
    )

    df["clicked"] = np.random.binomial(1, df["final_ctr"])
    df["converted"] = np.random.binomial(1, df["final_conversion_rate"])

    df["revenue"] = np.where(
        df["converted"] == 1,
        np.random.normal(df["avg_order_value"], df["avg_order_value"] * 0.20),
        0
    ).clip(min=0)

    df["strategy_name"] = experiment_brief["strategy_name"]
    df["primary_kpi"] = experiment_brief["primary_kpi"]

    return df


# -----------------------------
# Phase 2: Analyze A/B Test
# -----------------------------

def summarize_ab_test(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("experiment_group").agg(
        users=("user_id", "count"),
        clicks=("clicked", "sum"),
        conversions=("converted", "sum"),
        total_revenue=("revenue", "sum"),
        revenue_per_user=("revenue", "mean")
    ).reset_index()

    summary["ctr"] = summary["clicks"] / summary["users"]
    summary["conversion_rate"] = summary["conversions"] / summary["users"]
    summary["aov"] = summary["total_revenue"] / summary["conversions"]

    return summary


def run_significance_test(df: pd.DataFrame) -> dict:
    grouped = df.groupby("experiment_group")["converted"].agg(["sum", "count"])

    treatment_conversions = grouped.loc["treatment", "sum"]
    treatment_users = grouped.loc["treatment", "count"]
    control_conversions = grouped.loc["control", "sum"]
    control_users = grouped.loc["control", "count"]

    z_stat, p_value = proportions_ztest(
        count=np.array([treatment_conversions, control_conversions]),
        nobs=np.array([treatment_users, control_users]),
        alternative="larger"
    )

    treatment_rate = treatment_conversions / treatment_users
    control_rate = control_conversions / control_users

    return {
        "control_conversion_rate": control_rate,
        "treatment_conversion_rate": treatment_rate,
        "absolute_lift": treatment_rate - control_rate,
        "relative_lift": (treatment_rate - control_rate) / control_rate,
        "z_stat": z_stat,
        "p_value": p_value
    }


def analyze_segment_uplift(df: pd.DataFrame) -> pd.DataFrame:
    segment_summary = df.groupby(["customer_segment", "experiment_group"]).agg(
        users=("user_id", "count"),
        conversions=("converted", "sum"),
        revenue=("revenue", "sum")
    ).reset_index()

    segment_summary["conversion_rate"] = segment_summary["conversions"] / segment_summary["users"]
    segment_summary["revenue_per_user"] = segment_summary["revenue"] / segment_summary["users"]

    pivot = segment_summary.pivot(
        index="customer_segment",
        columns="experiment_group",
        values=["conversion_rate", "revenue_per_user"]
    )

    pivot.columns = ["_".join(col).strip() for col in pivot.columns]
    pivot = pivot.reset_index()

    pivot["conversion_relative_lift"] = (
        pivot["conversion_rate_treatment"] - pivot["conversion_rate_control"]
    ) / pivot["conversion_rate_control"]

    pivot["revenue_per_user_relative_lift"] = (
        pivot["revenue_per_user_treatment"] - pivot["revenue_per_user_control"]
    ) / pivot["revenue_per_user_control"]

    return pivot


# -----------------------------
# Phase 2: Visuals
# -----------------------------

def plot_conversion_rate(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(summary["experiment_group"], summary["conversion_rate"])
    plt.title("A/B Test Conversion Rate: Control vs AI-Agent Treatment")
    plt.xlabel("Experiment Group")
    plt.ylabel("Conversion Rate")
    plt.show()


def plot_segment_lift(segment_results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(segment_results["customer_segment"], segment_results["conversion_relative_lift"])
    plt.title("Conversion Lift by Customer Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Relative Conversion Lift")
    plt.xticks(rotation=30)
    plt.show()


# -----------------------------
# Run Full Phase 1 + Phase 2 Workflow
# -----------------------------

def main():
    print("=== Local Ecommerce Growth Agent: Phase 1 + Phase 2 Experimentation Engine ===\n")

    site_type = input("Website type, e.g. fashion, beauty, food, home goods, B2B supplies: ")
    product_type = input("Product type: ")
    customer_type = input("Customer type: ")
    goal = input("Business goal: ")
    issue = input("Current issue: ")
    budget = input("Budget: ")

    prompt = build_prompt(site_type, product_type, customer_type, goal, issue, budget)

    response = chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n=== PHASE 1: ECOMMERCE AGENT OUTPUT ===\n")
    print(response.message.content)

    experiment_brief = build_experiment_brief(site_type, product_type, customer_type, goal, issue, budget)

    print("\n=== PHASE 2: AGENT-GENERATED EXPERIMENT BRIEF ===\n")
    for key, value in experiment_brief.items():
        print(f"{key}: {value}")

    experiment_df = generate_experiment_data(experiment_brief)
    summary = summarize_ab_test(experiment_df)
    test_results = run_significance_test(experiment_df)
    segment_results = analyze_segment_uplift(experiment_df)

    print("\n=== PHASE 2: A/B TEST SUMMARY ===\n")
    print(summary)

    print("\n=== PHASE 2: SIGNIFICANCE TEST ===\n")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")

    print("\n=== PHASE 2: SEGMENT-LEVEL UPLIFT ===\n")
    print(segment_results)

    plot_conversion_rate(summary)
    plot_segment_lift(segment_results)


if __name__ == "__main__":
    main()
