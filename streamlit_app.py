import streamlit as st
import os
from call_nemotron import chat_with_text, invoke_url, kApiKey


st.set_page_config(page_title="BudgEat - Preferences", page_icon="🥗", layout="centered")

st.title("BudgEat")
st.subheader("Tell us your goals and preferences")

with st.form("preferences_form", clear_on_submit=False):
    weekly_budget = st.number_input(
        "Desired weekly food budget ($)",
        min_value=0.0,
        step=1.0,
        format="%.2f",
        help="Total amount you want to spend on food per week."
    )

    daily_calories = st.number_input(
        "Desired daily calorie intake", min_value=0, step=50, help="Target calories per day."
    )

    preferences = st.text_area(
        "Food preferences",
        placeholder="e.g., Mediterranean, vegan, high-protein, quick meals, spicy, gluten-free",
        help="Share cuisines, dietary styles, dislikes, or any constraints."
    )

    submitted = st.form_submit_button("Save Preferences")

if submitted:
    st.success("Preferences saved")
    st.write(
        {
            "weekly_budget_usd": weekly_budget,
            "daily_calories": daily_calories,
            "preferences": preferences.strip(),
        }
    )

    prompt = (
        "Using the following user goals, suggest a brief plan for budget-friendly, "
        "nutritious meals and shopping guidance for one week. "
        f"Weekly budget: ${weekly_budget:.2f}. "
        f"Daily calories: {int(daily_calories)} kcal. "
        f"Preferences: {preferences.strip() or 'None specified'}. "
        "Focus on variety, affordability, and practicality."
    )

    # Allow overriding API key via environment variable at runtime
    env_key = os.getenv("NVIDIA_API_KEY")
    if env_key and env_key != kApiKey:
        # Late binding into the imported module variable
        import call_nemotron as _cn
        _cn.kApiKey = env_key

    with st.spinner("Generating suggestions..."):
        try:
            resp = chat_with_text(infer_url=invoke_url, query=prompt, stream=False)
            # Try to extract assistant message
            content = None
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
            if content:
                st.markdown("## Suggested plan")
                st.write(content)
            else:
                st.markdown("## Raw response")
                st.write(resp)
        except Exception as e:
            st.error(f"Failed to generate response: {e}")
else:
    st.info("Fill in the form and click Save Preferences.")


