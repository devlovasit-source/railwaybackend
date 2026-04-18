def build_responsibility_map(event):
    tasks = []

    group = event.get("group")
    subtype = event.get("subtype")

    if group in ["kids", "school"]:
        tasks += ["pickup", "drop"]

    if group == "social":
        tasks.append("gift")

    if group == "finance":
        tasks.append("payment")

    if group == "travel":
        tasks += ["documents", "travel_prep", "airport_cab"]

    if subtype == "annual_day":
        tasks.append("costume_prep")

    return list(set(tasks))


def generate_family_prompts(event, responsibilities):
    prompts = []

    if "pickup" in responsibilities or "drop" in responsibilities:
        prompts.append("Who’s handling pickup and drop?")

    if "payment" in responsibilities:
        prompts.append("Want to assign who handles this payment?")

    if "gift" in responsibilities:
        prompts.append("Do you want to assign someone for the gift?")

    if event.get("group") == "travel":
        prompts.append("Want me to split travel prep between people?")

    return prompts