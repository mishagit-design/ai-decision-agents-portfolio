from ollama import chat


def public_health_levers(health_topic: str, population: str, region: str, goal: str, issue: str, data_available: str) -> str:
    text = f"{health_topic} {population} {region} {goal} {issue} {data_available}".lower()

    levers = []

    if "diabetes" in text or "chronic" in text:
        levers.append("Identify care gaps such as missed screenings, A1C monitoring gaps, medication refill gaps, and preventable ER visits")

    if "appointment" in text or "follow-up" in text or "no-show" in text:
        levers.append("Prioritize outreach for patients with missed appointments, delayed follow-ups, or repeated no-shows")

    if "access" in text or "transportation" in text or "low-income" in text or "underserved" in text:
        levers.append("Include social determinants of health such as transportation, income, insurance access, food access, and distance to care")

    if "screening" in text or "prevention" in text:
        levers.append("Segment the population by preventive screening status and prioritize high-risk groups for outreach")

    if "opioid" in text or "overdose" in text:
        levers.append("Monitor overdose risk indicators, prior utilization patterns, prescription history, and community-level risk factors")

    if "maternal" in text or "pregnancy" in text:
        levers.append("Track prenatal visit gaps, postpartum follow-up, high-risk conditions, and access barriers")

    if "outbreak" in text or "infectious" in text or "flu" in text or "covid" in text:
        levers.append("Monitor case trends, geographic clusters, vaccination gaps, and early-warning indicators")

    if "equity" in text or "disparity" in text:
        levers.append("Compare outcomes across population groups to identify inequities in access, prevention, and care completion")

    if not levers:
        levers = [
            "Define the target population and public health outcome clearly",
            "Identify risk signals, care gaps, and access barriers",
            "Segment the population by need, urgency, and intervention opportunity",
            "Recommend measurable interventions and track outcome improvement"
        ]

    return "\n".join(f"- {lever}" for lever in levers)


def build_prompt(health_topic: str, population: str, region: str, goal: str, issue: str, data_available: str) -> str:
    levers = public_health_levers(health_topic, population, region, goal, issue, data_available)

    return f"""
You are a public health data scientist with strong analytics, healthcare operations, and CDC-style research instincts.

Public health context:
Health topic: {health_topic}
Target population: {population}
Region or setting: {region}
Public health goal: {goal}
Current issue or bottleneck: {issue}
Available data: {data_available}

Suggested public health levers:
{levers}

Please provide:
1. A short diagnosis of what may be happening
2. The key risk signals or care gaps to investigate
3. The most relevant population segments to prioritize
4. Recommended intervention or outreach strategies
5. Suggested data features to engineer
6. Recommended KPIs or public health outcome measures
7. Equity, privacy, and ethical considerations
8. A simple 30-day action plan

Make the answer realistic for a public health, healthcare analytics, or CDC-oriented use case.
Keep it practical, specific, evidence-minded, and analytics-focused.
Do not provide medical advice. Focus on data science, population health, surveillance, intervention prioritization, and outcome measurement.
""".strip()


def main():
    print("=== Local Public Health Decision Intelligence Agent ===\n")

    health_topic = input("Health topic, e.g. diabetes, opioid overdose, maternal health, flu, preventive screening: ")
    population = input("Target population, e.g. adults 45+, rural communities, pregnant patients, high-risk patients: ")
    region = input("Region or setting, e.g. urban county, rural clinic network, state health department: ")
    goal = input("Public health goal: ")
    issue = input("Current issue or bottleneck: ")
    data_available = input("Available data, e.g. appointments, screenings, claims, ER visits, demographics, SDOH: ")

    prompt = build_prompt(
        health_topic,
        population,
        region,
        goal,
        issue,
        data_available
    )

    response = chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print("\n=== PUBLIC HEALTH AGENT OUTPUT ===\n")
    print(response.message.content)


if __name__ == "__main__":
    main()