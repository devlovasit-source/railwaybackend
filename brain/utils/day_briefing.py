from datetime import datetime


def format_time(iso: str):
    try:
        return datetime.fromisoformat(iso).strftime("%I:%M %p")
    except:
        return ""


# =========================
# SECTIONS
# =========================

def build_critical_section(results):
    lines = [
        f"{r['classifiedEvent']['title']} at {format_time(r['classifiedEvent']['startAtISO'])}"
        for r in results
        if r["classifiedEvent"].get("priority") == "critical"
    ]

    return {"label": "Critical today", "lines": lines}


def build_timed_section(results):
    sorted_results = sorted(results, key=lambda r: r["classifiedEvent"]["startAtISO"])

    lines = [
        f"{format_time(r['classifiedEvent']['startAtISO'])} · {r['classifiedEvent']['title']}"
        for r in sorted_results
    ]

    return {"label": "Today", "lines": lines}


def build_leave_by_section(results):
    lines = []

    for r in results:
        leave = r.get("predictiveOutput", {}).get("bufferPlan", {}).get("leaveByISO")
        if leave:
            lines.append(f"{r['classifiedEvent']['title']} · leave by {format_time(leave)}")

    return {"label": "Leave by", "lines": lines}


def build_payment_section(results):
    lines = []

    for r in results:
        if r["classifiedEvent"].get("group") == "finance":
            amount = r["classifiedEvent"].get("amount")
            text = r["classifiedEvent"]["title"]
            if amount:
                text += f" · ₹{amount}"
            lines.append(text)

    return {"label": "Due payments", "lines": lines}


def build_style_section(results):
    for r in results:
        outfit = r.get("predictiveOutput", {}).get("outfitPrompt")
        if outfit:
            return {
                "label": "Style hint",
                "lines": [
                    f"{r['classifiedEvent']['title']} · {', '.join(outfit.get('outfitKeywords', []))}"
                ]
            }
    return {"label": "Style hint", "lines": []}


def build_carry_section(results):
    for r in results:
        items = r.get("predictiveOutput", {}).get("packingList", [])
        if items:
            return {
                "label": "Carry hint",
                "lines": [
                    f"{r['classifiedEvent']['title']} · {', '.join(items[:4])}"
                ]
            }
    return {"label": "Carry hint", "lines": []}


def build_prep_section(results):
    lines = []

    for r in results:
        prep = r.get("checklistBundle", {}).get("prepTonight", {}).get("items", [])
        for item in prep[:3]:
            lines.append(f"{r['classifiedEvent']['title']} · {item}")

    return {"label": "Prep tonight", "lines": lines}


# =========================
# MAIN BUILDERS
# =========================

def build_morning_briefing(results):
    sections = [
        build_critical_section(results),
        build_timed_section(results),
        build_leave_by_section(results),
        build_payment_section(results),
        build_style_section(results),
        build_carry_section(results),
    ]

    return {
        "type": "morning_brief",
        "sections": [s for s in sections if s["lines"]]
    }


def build_evening_briefing(results):
    sections = [
        build_critical_section(results),
        build_prep_section(results),
        build_style_section(results),
        build_carry_section(results),
        build_payment_section(results),
    ]

    return {
        "type": "evening_prep_brief",
        "sections": [s for s in sections if s["lines"]]
    }


def build_busy_day_rescue(results):
    sorted_results = sorted(
        results,
        key=lambda r: r.get("predictiveOutput", {}).get("stressLoadScore", 0),
        reverse=True
    )

    return {
        "type": "busy_day_rescue",
        "sections": [
            {
                "label": "Top three must-not-miss",
                "lines": [
                    f"{r['classifiedEvent']['title']} at {format_time(r['classifiedEvent']['startAtISO'])}"
                    for r in sorted_results[:3]
                ]
            },
            {
                "label": "Prep first",
                "lines": [
                    f"{r['classifiedEvent']['title']} · {task}"
                    for r in sorted_results[:2]
                    for task in r.get("predictiveOutput", {}).get("prepTasks", [])[:2]
                ]
            }
        ]
    }


def build_best_day_briefing(results):
    stress = sum(r.get("predictiveOutput", {}).get("stressLoadScore", 0) for r in results)

    if stress >= 140 or len(results) >= 5:
        return build_busy_day_rescue(results)

    return build_morning_briefing(results)