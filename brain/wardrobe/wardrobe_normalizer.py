{
  "wardrobe_tag_normalizer_v1": {
    "version": "1.0",
    "description": "Normalizes wardrobe items into consistent tags so Shopping + Styling engines can match items reliably. Converts lens outputs and manual inputs into a compact 'wardrobe_index' used for outfit combos, compatibility scoring, and purchase conviction.",
    "inputs_supported": [
      "lens_item (detected category + attributes + raw_label)",
      "manual_item (user-entered category + description)",
      "imported_item (marketplace receipts / past purchases, optional)"
    ],
    "output_schema": {
      "wardrobe_index_item": {
        "wardrobe_id": "string",
        "title": "string",
        "gender_scope": { "type": "enum", "allowed": ["woman", "man", "unisex", "unknown"] },
        "category_group": "string",
        "subcategory": "string",
        "slots": ["string"],
        "colors": {
          "primary": "string",
          "secondary": ["string"],
          "family": { "type": "enum", "allowed": ["neutral", "warm", "cool", "bright", "earth", "pastel", "unknown"] }
        },
        "pattern": { "type": "enum", "allowed": ["solid", "striped", "floral", "geometric", "animal", "check", "polka", "abstract", "other", "unknown"] },
        "material": { "type": "enum", "allowed": ["cotton", "linen", "denim", "wool", "leather", "synthetic", "silk", "satin", "knit", "unknown"] },
        "finish": { "type": "enum", "allowed": ["matte", "glossy", "textured", "metallic", "unknown"] },
        "formality": { "type": "enum", "allowed": ["casual", "smart_casual", "semi_formal", "formal", "festive", "athleisure", "unknown"] },
        "season_tags": ["string"],
        "style_tags": ["string"],
        "fit_tags": ["string"],
        "occasion_tags": ["string"],
        "place_tags": ["string"],
        "compatibility": {
          "pairing_slots": ["string"],
          "avoid_with": ["string"]
        }
      }
    },
    "normalization_pipeline": [
      "map_category_group",
      "map_subcategory",
      "assign_slots",
      "normalize_colors",
      "normalize_pattern",
      "normalize_material_finish",
      "infer_formality",
      "infer_season_tags",
      "infer_style_tags",
      "infer_fit_tags",
      "infer_occasion_place_tags",
      "generate_pairing_slots",
      "generate_avoid_with_rules",
      "return_normalized_item"
    ],
    "category_maps": {
      "category_group_map": {
        "top": ["t-shirt", "shirt", "blouse", "tank", "crop_top", "kurta_top"],
        "bottom": ["jeans", "trousers", "pants", "skirt", "shorts", "palazzo"],
        "dress": ["dress", "gown", "maxi_dress"],
        "layer": ["blazer", "jacket", "coat", "cardigan", "overshirt", "shrug"],
        "shoe": ["sneaker", "loafer", "heel", "sandal", "boot", "flat"],
        "bag": ["tote", "satchel", "crossbody", "shoulder_bag", "clutch", "backpack"],
        "accessory": ["belt", "sunglasses", "scarf", "hat"],
        "jewelry": ["earrings", "necklace", "bracelet", "ring"],
        "indian_wear": ["saree", "lehenga", "salwar", "kurta_set", "dupatta"]
      },
      "raw_label_to_subcategory_map": {
        "short_sleeved_shirt": "t-shirt",
        "long_sleeved_shirt": "shirt",
        "dress": "dress",
        "blazer": "blazer",
        "jacket": "jacket",
        "jeans": "jeans",
        "trousers": "trousers",
        "skirt": "skirt",
        "sneaker": "sneaker",
        "handbag": "shoulder_bag"
      }
    },
    "slot_rules": {
      "slots_by_category_group": {
        "top": ["top_base"],
        "bottom": ["bottom_base"],
        "dress": ["one_piece_base"],
        "layer": ["outer_layer"],
        "shoe": ["shoe"],
        "bag": ["bag"],
        "accessory": ["accessory"],
        "jewelry": ["jewelry"],
        "indian_wear": ["ethnic_base"]
      },
      "special_slot_overrides": [
        {
          "if_subcategory_in": ["blazer", "coat", "jacket"],
          "add_slots": ["structure_layer"]
        },
        {
          "if_subcategory_in": ["belt"],
          "add_slots": ["waist_definition"]
        }
      ]
    },
    "color_normalization": {
      "primary_color_priority": ["black", "white", "cream", "beige", "tan", "brown", "navy", "blue", "grey", "red", "pink", "green", "yellow", "orange", "purple", "silver", "gold"],
      "families": {
        "neutral": ["black", "white", "cream", "beige", "tan", "brown", "grey", "navy"],
        "warm": ["red", "orange", "rust", "coral", "mustard", "gold"],
        "cool": ["blue", "green", "teal", "purple", "silver"],
        "bright": ["hot_pink", "neon_green", "bright_yellow"],
        "earth": ["olive", "khaki", "camel", "terracotta"],
        "pastel": ["baby_pink", "mint", "lavender", "powder_blue"]
      },
      "notes": [
        "If multiple colors exist, choose primary as the most dominant or most neutral anchor.",
        "If unknown, set primary='unknown' and family='unknown'."
      ]
    },
    "pattern_normalization": {
      "keywords_to_pattern": [
        { "match_any": ["stripe", "striped"], "pattern": "striped" },
        { "match_any": ["check", "checked", "plaid"], "pattern": "check" },
        { "match_any": ["polka"], "pattern": "polka" },
        { "match_any": ["floral"], "pattern": "floral" },
        { "match_any": ["animal", "leopard", "zebra"], "pattern": "animal" },
        { "match_any": ["geo", "geometric"], "pattern": "geometric" }
      ],
      "default": "solid"
    },
    "material_finish_normalization": {
      "material_keywords": [
        { "match_any": ["denim"], "material": "denim" },
        { "match_any": ["cotton"], "material": "cotton" },
        { "match_any": ["linen"], "material": "linen" },
        { "match_any": ["wool"], "material": "wool" },
        { "match_any": ["leather", "pu"], "material": "leather" },
        { "match_any": ["silk"], "material": "silk" },
        { "match_any": ["satin"], "material": "satin" },
        { "match_any": ["knit"], "material": "knit" }
      ],
      "finish_keywords": [
        { "match_any": ["gloss", "shiny", "patent"], "finish": "glossy" },
        { "match_any": ["matte"], "finish": "matte" },
        { "match_any": ["textured", "woven", "ribbed"], "finish": "textured" },
        { "match_any": ["metallic"], "finish": "metallic" }
      ],
      "defaults": { "material": "unknown", "finish": "unknown" }
    },
    "formality_inference": {
      "rules": [
        {
          "if_any": { "subcategory_in": ["blazer", "trousers", "loafer"], "pattern_not": ["animal"] },
          "set_formality": "smart_casual"
        },
        {
          "if_any": { "subcategory_in": ["gown"], "finish_in": ["glossy", "metallic"] },
          "set_formality": "formal"
        },
        {
          "if_any": { "subcategory_in": ["sneaker", "t-shirt", "jeans"] },
          "set_formality": "casual"
        },
        {
          "if_any": { "category_group_is": "indian_wear" },
          "set_formality": "festive"
        },
        {
          "fallback": true,
          "set_formality": "unknown"
        }
      ]
    },
    "season_inference": {
      "season_tags": ["india_summer", "india_monsoon", "india_winter", "all_season"],
      "rules": [
        { "if_material_in": ["linen"], "add": ["india_summer"] },
        { "if_material_in": ["wool", "knit"], "add": ["india_winter"] },
        { "if_subcategory_in": ["boot"], "add": ["india_winter"] },
        { "if_subcategory_in": ["sandal"], "add": ["india_summer"] },
        { "if_material_in": ["denim", "cotton"], "add": ["all_season"] }
      ],
      "fallback": ["all_season"]
    },
    "style_tag_inference": {
      "rules": [
        { "if_pattern_in": ["solid"], "add": ["minimal", "classic"] },
        { "if_pattern_in": ["animal"], "add": ["edgy", "glam"] },
        { "if_finish_in": ["metallic", "glossy"], "add": ["glam"] },
        { "if_subcategory_in": ["sneaker", "overshirt"], "add": ["street"] },
        { "if_category_group_is": "indian_wear", "add": ["ethnic_traditional", "fusion"] }
      ],
      "notes": [
        "These are soft tags for recommendation matching; do not show them to the user."
      ]
    },
    "fit_tag_inference": {
      "rules": [
        { "if_title_contains_any": ["oversized"], "add": ["relaxed_fit"] },
        { "if_title_contains_any": ["slim", "skinny"], "add": ["slim_fit"] },
        { "if_title_contains_any": ["cropped"], "add": ["cropped_length"] },
        { "if_title_contains_any": ["high waist", "high-waist"], "add": ["high_waist"] }
      ],
      "defaults": []
    },
    "occasion_place_inference": {
      "occasion_tags": ["work_day", "meeting_day", "interview", "date_night", "brunch", "wedding_guest", "party_night_out", "airport_travel", "vacation_resort", "festive_ethnic", "casual_everyday", "shopping_day"],
      "place_tags": ["office", "cafe_brunch", "club_night", "wedding_venue", "airport", "mall_shopping", "beach_resort", "home_hosting", "temple_religious", "travel_city_walk"],
      "rules": [
        { "if_subcategory_in": ["blazer", "trousers", "loafer"], "add_occasion": ["work_day", "meeting_day"], "add_place": ["office"] },
        { "if_subcategory_in": ["clutch"], "add_occasion": ["party_night_out", "wedding_guest"], "add_place": ["club_night", "wedding_venue"] },
        { "if_subcategory_in": ["sneaker", "backpack"], "add_occasion": ["airport_travel", "shopping_day"], "add_place": ["airport", "mall_shopping"] },
        { "if_category_group_is": "indian_wear", "add_occasion": ["festive_ethnic", "wedding_guest"], "add_place": ["home_hosting", "wedding_venue", "temple_religious"] }
      ]
    },
    "pairing_slot_generation": {
      "pairing_slots_map": {
        "top_base": ["bottom_base", "outer_layer", "shoe", "bag", "jewelry", "accessory"],
        "bottom_base": ["top_base", "outer_layer", "shoe", "bag", "jewelry", "accessory"],
        "one_piece_base": ["outer_layer", "shoe", "bag", "jewelry", "accessory"],
        "ethnic_base": ["outer_layer", "shoe", "bag", "jewelry", "accessory"],
        "outer_layer": ["top_base", "bottom_base", "one_piece_base", "shoe", "bag", "jewelry"],
        "shoe": ["top_base", "bottom_base", "one_piece_base", "ethnic_base", "outer_layer", "bag"],
        "bag": ["top_base", "bottom_base", "one_piece_base", "ethnic_base", "shoe"],
        "jewelry": ["top_base", "one_piece_base", "ethnic_base"],
        "accessory": ["top_base", "one_piece_base", "ethnic_base"]
      }
    },
    "avoid_with_rules": {
      "default": [],
      "rules": [
        {
          "if_pattern_in": ["animal"],
          "add_avoid_with": ["pattern=animal", "pattern=geometric", "pattern=check"],
          "notes": "Prevent print clashes unless user print tolerance is high."
        },
        {
          "if_finish_in": ["metallic"],
          "add_avoid_with": ["finish=metallic"],
          "notes": "Avoid stacking too many shiny hero elements by default."
        }
      ]
    },
    "compatibility_scoring": {
      "description": "Used by purchase conviction engine to pick best wardrobe anchors.",
      "weights": {
        "category_slot_match": 0.35,
        "color_harmony_match": 0.25,
        "formality_match": 0.2,
        "season_match": 0.1,
        "style_tag_match": 0.1
      },
      "score_range": [0, 1],
      "thresholds": {
        "strong_match": 0.7,
        "okay_match": 0.45,
        "weak_match": 0.25
      }
    },
    "dev_notes": [
      "This normalizer is the glue between lens outputs and outfit generation.",
      "Keep tags internal; do not show raw tags to users.",
      "If uncertain about a tag, set 'unknown' instead of guessing.",
      "When wardrobe data is missing, shopping engine uses generic staples but still outputs 3 combos."
    ]
  }
}