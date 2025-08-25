import rjsmin
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # this is app/server/webpage
NOTIF_JS_PATH = BASE_DIR / "notification.js"
INFO_BOX_PATH = BASE_DIR / "info_box.html"


user = "ferdinand"
domain = "koenix.de"
obfuscated_email = f"{user} &lt;at&gt; {domain}"

# Load info box from HTML file
with open(INFO_BOX_PATH, "r", encoding="utf-8") as f:
    info_box_html = f.read()

# Replace placeholder for email if present
info_box_html = info_box_html.replace("{obfuscated_email}", obfuscated_email)



def spinner_html(message: str) -> str:
    return f"""
    <div id="insight-spinner" style="display: flex; align-items: center; gap: 8px;">
        <div style="
            flex: none; 
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 1s linear infinite;">
        </div>
        <span>{message}</span>
    </div>
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """


# Read the JS code
with open(NOTIF_JS_PATH, mode="r", encoding="utf-8") as f:
    js_code = f.read()

# Minify using rjsmin
notification_js = rjsmin.jsmin(js_code)