import requests
import os
from dotenv import load_dotenv

# ────────────────────────────────────────────────
# CONFIGURATION - CHANGE THESE VALUES
# ────────────────────────────────────────────────

API_VERSION = "v24.0"  # Latest as of Jan 2026
GRAPH_URL = f"https://graph.facebook.com/{API_VERSION}"
GRAPH_VIDEO_URL = f"https://graph-video.facebook.com/{API_VERSION}"  # Still used for video uploads


load_dotenv()
PAGE_ID = os.getenv("GEMINI_API_KEY")
PAGE_ACCESS_TOKEN =  os.getenv("GEMINI_API_KEY")

# ────────────────────────────────────────────────
# Helper function to handle API response
# ────────────────────────────────────────────────
def handle_response(response):
    if response.status_code == 200:
        data = response.json()
        if "id" in data or "post_id" in data:
            print("Success!")
            print("Post/Video/Photo ID:", data.get("id") or data.get("post_id"))
            if "post_id" in data:
                print("Associated Post ID:", data["post_id"])
            return True
        else:
            print("Strange success response:", data)
            return False
    else:
        print("Error! Status code:", response.status_code)
        try:
            error_data = response.json()
            print("Facebook Error:", error_data.get("error", {}).get("message", "Unknown error"))
            print("Full error:", error_data)
        except:
            print("Response text:", response.text)
        return False

# ────────────────────────────────────────────────
# 1. Post plain text (message/description only)
# ────────────────────────────────────────────────
def post_text(message):
    print("\nPosting text...")
    url = f"{GRAPH_URL}/{PAGE_ID}/feed"
    
    payload = {
        "message": message,
        "access_token": PAGE_ACCESS_TOKEN,
        "published": "true"
    }
    
    response = requests.post(url, data=payload)
    return handle_response(response)

# ────────────────────────────────────────────────
# 2. Post photo with caption
# ────────────────────────────────────────────────
def post_photo(photo_source, caption=""):
    print("\nPosting photo...")
    url = f"{GRAPH_URL}/{PAGE_ID}/photos"

    data = {
        "caption": caption,          # <-- use caption for /photos
        "access_token": PAGE_ACCESS_TOKEN,
        "published": "true"
    }

    if photo_source.startswith("http"):
        data["url"] = photo_source
        response = requests.post(url, data=data)
        return handle_response(response)

    if not os.path.exists(photo_source):
        print(f"File not found: {photo_source}")
        return False

    with open(photo_source, "rb") as f:
        files = {"source": f}
        response = requests.post(url, data=data, files=files)
        return handle_response(response)


# ────────────────────────────────────────────────
# 3. Post video with title & description
# ────────────────────────────────────────────────
def post_video(video_path, title="", description=""):
    print("\nPosting video...")
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False

    url = f"{GRAPH_VIDEO_URL}/{PAGE_ID}/videos"

    data = {
        "title": title,
        "description": description,
        "access_token": PAGE_ACCESS_TOKEN,
        "published": "true"
        # Optional: "published": "false"  # draft/unpublished
    }

    # IMPORTANT: Force a short filename so Facebook doesn't receive the full path
    with open(video_path, "rb") as f:
        files = {
            "source": ("video.mp4", f, "video/mp4")  # <- overrides filename + sets MIME
        }
        response = requests.post(url, data=data, files=files)

    return handle_response(response)


# ────────────────────────────────────────────────
# MAIN - Uncomment the example you want to run
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Example 1: Text post
    # post_text("Test post from Python using Graph API v24.0!\nDelhi vibes 🌆 #Automation")

    # Example 2: Photo from URL
    # post_photo(
    #     photo_source="https://example.com/your-image.jpg",
    #     caption="Stunning view from Delhi! 📸 Posted via v24.0 script."
    # )

    # Example 3: Local photo
    post_photo(
        photo_source="/Users/pragyan/Voicecline/app/asasas.png",
        caption="संघ लोक सेवा आयोग (UPSC) ने नियमों में बड़ा बदलाव किया है। UPSC की तरफ से जारी नई गाइडलाइन्स में कहा गया है कि पहले से सेलेक्टेड उम्मीदवार बार-बार परीक्षा नहीं दे पाएंगे। " \
        "एलिजिबिलिटी से जुड़े दिशानिर्देश में कहा गया है कि पहले से ही IAS या IFS अधिकारी के रूप में नियुक्त या चुने गए उम्मीदवारों को दूसरा मौका नहीं दिया जाएगा। भारतीय पुलिस सेवा (IPS) में चुने गए या नियुक्त लोगों के लिए भी ऐसा ही नियम है, जिसका मतलब है कि वे CSE 2026 के लिए एलिजिबल नहीं होंगे। UPSC ने सख्त नियम बनाए हैं, जिसमें IAS या IFS अधिकारियों के दोबारा प्रयास पर रोक लगाई गई है। 4 फरवरी के सर्कुलर में कहा गया है, जो उम्मीदवार पिछली परीक्षा के नतीजों के आधार पर भारतीय प्रशासनिक सेवा (IAS) या भारतीय विदेश सेवा (IFS) में नियुक्त हुआ है और उस सेवा का सदस्य बना हुआ है। वह सिविल सेवा परीक्षा-2026 में शामिल होने के लिए एलिजिबल नहीं होगा। "
    )

    # Example 4: Local video (recommended first test)
    # post_video(
    #     video_path="/Users/pragyan/Voicecline/app/asasasasa.jpeg",         # Update path
    #     title="धार जिले की मनावर तहसील",
    #     description="पुलिस ने इस मामले में ब्लॉक इंचार्ज मुरली कृष्णा और सहायक महादेव यादव के अलावा फर्राना मशीन के चालक के खिलाफ भारतीय न्याय संहिता (बीएनएस) की धारा 125(ए) और 125(बी) के तहत मामला दर्ज किया है। "
    # )

    print("\nScript finished.")