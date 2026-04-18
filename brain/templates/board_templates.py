# backend/brain/templates/board_templates.py

AHVI_TEMPLATES = [

    # =========================
    # Mannequin — Top + Bottom
    # =========================
    {
        "id": "mannequin.topBottom",
        "title": "Clean Two-Piece Look",
        "roles": [
            {"id": "top", "required": True, "keywords": ["shirt","t-shirt","top","polo","blouse","knit","sweater"]},
            {"id": "bottom", "required": True, "keywords": ["jeans","pants","trousers","skirt","shorts","palazzo","wide leg"]},
            {"id": "shoes", "required": True, "keywords": ["sneakers","flats","heels","loafers","slides","sandals"]},
            {"id": "bag", "required": True, "keywords": ["shoulder","crossbody","sling","tote","mini","clutch"]},
            {"id": "earrings", "required": False, "keywords": ["earrings","studs","hoops","chandbali"]},
            {"id": "bracelet", "required": False, "keywords": ["bracelet","bangle","kada"]},
            {"id": "watch", "required": False, "keywords": ["watch"]},
            {"id": "sunglasses", "required": False, "keywords": ["sunglasses","shades"]}
        ],
        "layout": [
            {"role":"top", "x":4, "y":4, "w":52, "h":38, "z":2},
            {"role":"bottom", "x":58, "y":10, "w":38, "h":46, "z":2},

            {"role":"earrings", "x":76, "y":15, "w":12, "h":12, "z":4},
            {"role":"sunglasses", "x":69, "y":26, "w":20, "h":12, "z":4},
            {"role":"bracelet", "x":10, "y":54, "w":18, "h":12, "z":4},
            {"role":"watch", "x":10, "y":70, "w":12, "h":12, "z":4},
            {"role":"bag", "x":24, "y":63, "w":36, "h":18, "z":3},
            {"role":"shoes", "x":6, "y":82, "w":44, "h":16, "z":3}
        ]
    },

    # ======================
    # Mannequin — Dress
    # ======================
    {
        "id": "mannequin.dress",
        "title": "Clean One-Piece Look",
        "roles": [
            {"id": "dress", "required": True, "keywords": ["dress","gown","jumpsuit"]},
            {"id": "shoes", "required": True, "keywords": ["heels","flats","sandals","slides"]},
            {"id": "bag", "required": True, "keywords": ["clutch","shoulder","crossbody","tote","mini"]},
            {"id": "earrings", "required": False, "keywords": ["earrings","studs","hoops","chandbali"]},
            {"id": "necklace", "required": False, "keywords": ["necklace","pendant","choker"]},
            {"id": "sunglasses", "required": False, "keywords": ["sunglasses","shades"]}
        ],
        "layout": [
            {"role":"dress", "x":12, "y":2, "w":58, "h":64, "z":2},
            {"role":"earrings", "x":76, "y":10, "w":12, "h":12, "z":4},
            {"role":"necklace", "x":62, "y":28, "w":32, "h":18, "z":4},
            {"role":"shoes", "x":56, "y":74, "w":36, "h":16, "z":3},
            {"role":"bag", "x":60, "y":58, "w":30, "h":18, "z":3}
        ]
    },

    # ======================
    # Festive
    # ======================
    {
        "id": "mannequin.festive",
        "title": "Festive Set",
        "roles": [
            {"id":"outfit","required":True,"keywords":["lehenga","saree","kurta","gown","sherwani"]},
            {"id":"shoes","required":True,"keywords":["heels","juttis","sandals"]},
            {"id":"bag","required":True,"keywords":["clutch","potli","mini"]},
            {"id":"earrings","required":True,"keywords":["earrings","jhumka","chandbali"]},
            {"id":"necklace","required":False,"keywords":["necklace","choker"]},
            {"id":"bracelet","required":False,"keywords":["bangle","bracelet"]},
            {"id":"maang","required":False,"keywords":["maang tikka"]}
        ],
        "layout": [
            {"role":"outfit","x":22,"y":6,"w":56,"h":72,"z":2},
            {"role":"earrings","x":76,"y":10,"w":10,"h":10,"z":4},
            {"role":"necklace","x":12,"y":18,"w":18,"h":12,"z":4},
            {"role":"maang","x":50,"y":8,"w":10,"h":8,"z":4},
            {"role":"bag","x":70,"y":48,"w":22,"h":18,"z":3},
            {"role":"shoes","x":20,"y":80,"w":30,"h":16,"z":3},
            {"role":"bracelet","x":8,"y":52,"w":12,"h":12,"z":4}
        ]
    }
]