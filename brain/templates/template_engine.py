from brain.templates.board_templates import AHVI_TEMPLATES


def select_template(outfit_data):
    """
    Decide which template to use based on outfit type
    """
    if outfit_data.get("dress"):
        return next(t for t in AHVI_TEMPLATES if t["id"] == "mannequin.dress")

    if outfit_data.get("top") and outfit_data.get("bottom"):
        return next(t for t in AHVI_TEMPLATES if t["id"] == "mannequin.topBottom")

    return AHVI_TEMPLATES[0]


def match_items_to_roles(template, wardrobe, outfit_data):
    """
    Map selected outfit items to template roles
    """
    matched = []

    for role in template["roles"]:
        role_id = role["id"]

        # match from outfit_data (BEST)
        selected_name = outfit_data.get(role_id)

        for item in wardrobe:
            item_name = item.get("name", "")

            if selected_name and selected_name == item_name:
                matched.append({
                    "role": role_id,
                    "item": item
                })
                break

    return matched


def build_board(outfit_data, wardrobe):
    """
    Final board structure for frontend
    """
    template = select_template(outfit_data)
    items = match_items_to_roles(template, wardrobe, outfit_data)

    return {
        "template_id": template["id"],
        "layout": template["layout"],
        "items": items
    }