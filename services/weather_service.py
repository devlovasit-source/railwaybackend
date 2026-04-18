import requests
from datetime import datetime


def get_hourly_weather(lat: float, lon: float):

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,weathercode"
            f"&timezone=auto"
        )

        res = requests.get(url, timeout=5)
        data = res.json()

        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        codes = data["hourly"]["weathercode"]

        # current hour
        now_hour = datetime.now().hour

        # find closest hour index
        idx = now_hour

        temp = temps[idx]
        code = codes[idx]

        # -------------------------
        # TEMP LOGIC
        # -------------------------
        if temp > 30:
            condition = "hot"
        elif temp < 18:
            condition = "cold"
        else:
            condition = "mild"

        # -------------------------
        # RAIN LOGIC
        # -------------------------
        rain_codes = [51, 53, 55, 61, 63, 65, 80, 81, 82]

        if code in rain_codes:
            condition = "rainy"

        # -------------------------
        # TIME OF DAY 🔥
        # -------------------------
        if 5 <= now_hour < 12:
            time_of_day = "morning"
        elif 12 <= now_hour < 17:
            time_of_day = "afternoon"
        elif 17 <= now_hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        return {
            "temperature": temp,
            "condition": condition,
            "time_of_day": time_of_day
        }

    except Exception as e:
        print("Weather fetch failed:", str(e))
        return {
            "condition": "mild",
            "time_of_day": "day"
        }