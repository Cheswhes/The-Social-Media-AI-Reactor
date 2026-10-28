import json
import ollama

# CHANGE THIS to your exact model name from `ollama list`
MODEL = "gemma4"

# ---------- Load the post ----------

post = json.load(open("post.json", encoding="utf-8"))

context = (
    "CAPTION:\n" + post["caption"]
    + "\n\nCOMMENTS:\n" + "\n".join(post["comments"])
    + "\n\nMETADATA:\n" + json.dumps(post["metadata"], indent=2)
)

# ---------- Part 1: The 5 analysis layers ----------

layers = {
    "1_content_semantic": "Layer 1. Summarize the main message of the caption, list the key themes and the tone of the comments. If an image is attached, transcribe any overlay text and describe what it shows. Return JSON with keys: summary, themes, comment_overview, image_text, image_description.",
    "2_sentiment_emotion": "Layer 2. Classify sentiment of the caption and of the comments as positive, negative, neutral or mixed. Give the dominant emotions with a 0-1 score each and a one-line reason. Return JSON with keys: caption_sentiment, comments_sentiment, emotions, reason.",
    "3_entity_intent": "Layer 3. Extract people, organizations, brands, products, locations and events. Then identify the author's intent (inform, promote, recruit, brag, seek engagement, etc). Return JSON with keys: people, organizations, brands, products, locations, events, author_intent.",
    "4_content_metadata": "Layer 4. Combine content and metadata. Judge reach potential from follower count, engagement rate (likes+comments+reposts relative to followers), hashtag strategy, posting time, and whether the engagement fits the content. Return JSON with keys: engagement_rate_estimate, reach_assessment, hashtag_analysis, timing_note, content_metadata_fit.",
    "5_risk_safety": "Layer 5. Moderation check. Flag any hate speech, misinformation, spam or engagement bait, self-harm risk, copyright concerns, or LinkedIn policy violations. Return JSON with keys: flags (list of {type, severity, evidence}), overall_risk (low/medium/high), recommended_action.",
}

results = {}

for name, task in layers.items():
    msg = {
        "role": "user",
        "content": task
        + "\n\n" + context
        + "\n\nPREVIOUS LAYER RESULTS:\n" + json.dumps(results, ensure_ascii=False),
    }
    # only layers 1 and 5 get the image
    if post["image_path"] and name in ("1_content_semantic", "5_risk_safety"):
        msg["images"] = [post["image_path"]]

    r = ollama.chat(
        model=MODEL,
        messages=[msg],
        format="json",
        think=False,
        options={"temperature": 0.2},
    )
    results[name] = json.loads(r["message"]["content"])

    print("\n=== " + name + " ===")
    print(json.dumps(results[name], indent=2, ensure_ascii=False))

json.dump(results, open("analysis.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- Part 2: The 6 reaction comments ----------

human_rules = """You are a real LinkedIn user writing a quick comment on a post.
Rules:
- 1 to 3 sentences, under 45 words.
- Refer to one specific detail from the post, not the topic in general.
- Write casually, like typing on a phone. Contractions are fine. A tiny imperfection is fine.
- No hashtags. At most one emoji, and often none.
- Never use: "Great post", "Thanks for sharing", "game-changer", "delve", "insightful", "In today's world", "It's important to note".
- No bullet points, no summary of the post, no sign-off.
- Output only the comment text."""

categories = {
    "1. Value-Add": {
        "goal": "Demonstrate expertise & add data",
        "impact": "Very High (Drives profile views & authoritative replies)",
        "style": "Add one concrete fact, number, tool or framework that extends the post. Sound like someone with real experience.",
    },
    "2. Experience Share": {
        "goal": "Build trust & authentic rapport",
        "impact": "High (Fosters deeper peer connections)",
        "style": "Share a short personal story or result that relates to the post, with one specific detail.",
    },
    "3. Respectful Challenge": {
        "goal": "Spark debate & offer counterpoints",
        "impact": "High (Generates thoughtful thread activity)",
        "style": "Politely disagree with or complicate one point. Acknowledge what is right first, then give a counterpoint.",
    },
    "4. Discussion Starter": {
        "goal": "Gain knowledge & invite elaboration",
        "impact": "Medium-High (Directly engages the post author)",
        "style": "Ask the author one specific, curious question that invites them to elaborate.",
    },
    "5. Supportive / Social": {
        "goal": "Maintain network presence & goodwill",
        "impact": "Low-Medium (Sustains relationship touchpoints)",
        "style": "A short, warm, genuine reaction that congratulates or encourages, mentioning one detail.",
    },
    "6. The Disruptor (Troll)": {
        "goal": "Derail, undermine, or outrage-farm",
        "impact": "Volatile (Drives raw comment volume, but risks toxic sentiment)",
        "style": "Simulated for a moderation study: a dismissive, cynical, derailing comment of the kind trolls post. No slurs, no threats, no personal attacks on named individuals.",
    },
}

comments = {}

for name, info in categories.items():
    prompt = (
        "POST SUMMARY: " + json.dumps(results["1_content_semantic"], ensure_ascii=False)
        + "\n\nORIGINAL CAPTION:\n" + post["caption"]
        + "\n\nExisting comments (do not repeat their wording):\n" + "\n".join(post["comments"])
        + "\n\nWrite one comment."
        + "\nCategory: " + name
        + "\nPrimary goal: " + info["goal"]
        + "\nHow to write it: " + info["style"]
    )

    # Step A: first draft (high temperature = more varied wording)
    draft = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": human_rules},
            {"role": "user", "content": prompt},
        ],
        think=False,
        options={"temperature": 0.9, "top_p": 0.95},
    )["message"]["content"].strip()

    # Step B: rewrite so it sounds less polished
    final = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": human_rules},
            {
                "role": "user",
                "content": "Rewrite this comment so it sounds more like a real person typed it quickly. Keep the meaning and category. Remove any polished or generic phrasing. Output only the comment.\n\n" + draft,
            },
        ],
        think=False,
        options={"temperature": 0.8},
    )["message"]["content"].strip()

    # Step C: moderation check on the final comment
    check = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Moderation check on this LinkedIn comment. Return JSON with keys: flags, overall_risk (low/medium/high), would_remove (true/false).\n\n" + final,
            }
        ],
        format="json",
        think=False,
        options={"temperature": 0.1},
    )["message"]["content"]

    comments[name] = {
        "primary_goal": info["goal"],
        "engagement_impact": info["impact"],
        "draft": draft,
        "final": final,
        "moderation": json.loads(check),
    }

    print("\n=== " + name + " ===")
    print("GOAL:   " + info["goal"])
    print("IMPACT: " + info["impact"])
    print("DRAFT:  " + draft)
    print("FINAL:  " + final)
    print("MODERATION: " + json.dumps(comments[name]["moderation"], ensure_ascii=False))

json.dump(comments, open("comments.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("\nDone. Saved analysis.json and comments.json")