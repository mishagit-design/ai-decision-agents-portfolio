from ollama import chat

def ecommerce_levers(site_type: str, product_type: str, customer_type: str, goal: str, issue: str, budget: str) -> str:
    text = f"{site_type} {product_type} {customer_type} {goal} {issue} {budget}".lower()

    levers = []

    if "traffic" in text and ("low sales" in text or "low conversion" in text):
        levers.append("Check landing page relevance, product-page clarity, pricing, trust signals, and checkout friction")

    if "cart" in text or "abandon" in text:
        levers.append("Improve cart recovery: email/SMS reminders, shipping transparency, payment options, and urgency messaging")

    if "repeat" in text or "retention" in text:
        levers.append("Use lifecycle email, loyalty offers, replenishment reminders, and customer segmentation")

    if "new customer" in text or "acquisition" in text:
        levers.append("Focus on SEO, paid search, social proof, creator content, and first-purchase offers")

    if "average order" in text or "aov" in text:
        levers.append("Use bundles, thresholds for free shipping, cross-sells, upsells, and product recommendations")

    if "low" in text:
        levers.append("Prioritize low-cost improvements: SEO, email, product page optimization, reviews, and retargeting")

    if not levers:
        levers = [
            "Review the customer journey from traffic source to checkout",
            "Improve product pages with clearer benefits, images, reviews, FAQs, and trust signals",
            "Use email, retargeting, and segmentation to recover and nurture shoppers",
            "Track funnel metrics to identify the biggest drop-off point"
        ]

    return "\n".join(f"- {lever}" for lever in levers)


def build_prompt(site_type: str, product_type: str, customer_type: str, goal: str, issue: str, budget: str) -> str:
    levers = ecommerce_levers(site_type, product_type, customer_type, goal, issue, budget)

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
{levers}

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


def main():
    print("=== Local Ecommerce Growth Agent ===\n")

    site_type = input("Website type, e.g. fashion, beauty, food, home goods, B2B supplies: ")
    product_type = input("Product type: ")
    customer_type = input("Customer type: ")
    goal = input("Business goal: ")
    issue = input("Current issue: ")
    budget = input("Budget: ")

    prompt = build_prompt(site_type, product_type, customer_type, goal, issue, budget)

    response = chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print("\n=== ECOMMERCE AGENT OUTPUT ===\n")
    print(response.message.content)


if __name__ == "__main__":
    main()
