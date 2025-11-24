import praw

reddit = praw.Reddit(
    client_id="XCnVe87cDVFbnkuHo1GIKQ",
    client_secret="y7jV9PpsvZksT26-Oe71-oS5DA_-Kw",
    user_agent="testscript by u/No-Biscotti8980"
)

posts = reddit.subreddit("comics").top(limit=5)

for p in posts:
    print("Titel:", p.title)

    # direct image (imgur, i.redd.it, etc.)
    if getattr(p, "preview", None):
        for img in p.preview.get("images", []):
            print("image:", img["source"]["url"])

    # Reddit gallery posts
    if getattr(p, "is_gallery", False):
        for item in p.gallery_data["items"]:
            media_id = item["media_id"]
            print("gallery image:", p.media_metadata[media_id]["s"]["u"])

    # fallback to submission URL so you always see something
    if not getattr(p, "preview", None) and not getattr(p, "is_gallery", False):
        print("image/link:", p.url)

    print("-" * 50)
