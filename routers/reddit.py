import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

# We use a custom User-Agent so Reddit doesn't block our requests
HEADERS = {'User-Agent': 'AhviFashionBot/1.0'}
FASHION_SUBREDDITS = ["streetwear", "malefashionadvice", "femalefashionadvice"]

@router.get("/api/reddit-trends")
def get_fashion_trends(limit: int = 3):
    """Fetches the top current posts from fashion subreddits."""
    trends = []
    
    try:
        for sub in FASHION_SUBREDDITS:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title")
                text = post_data.get("selftext", "")
                
                # Only grab posts that actually have text or discussion
                if title and not post_data.get("stickied"):
                    trends.append({
                        "subreddit": sub,
                        "title": title,
                        "content": text[:200] + "..." if len(text) > 200 else text
                    })
                    
        return {
            "success": True,
            "message": "Fetched latest Reddit trends!",
            "trends": trends
        }
        
    except Exception as e:
        print(f"Reddit Fetch Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not reach Reddit servers.")