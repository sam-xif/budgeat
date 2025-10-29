import streamlit as st
import os
import json
from agent import ResearchAgent
from dotenv import load_dotenv

# Load environment variables from .env before importing call_nemotron
load_dotenv()

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
            resp = chat_with_text(
                infer_url=invoke_url, 
                query=prompt, 
                stream=False,
                force_json=True,
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Recipe name"
                            },
                            "ingredients": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "Ingredient name",
                                }
                            },
                        },
                        "required": ["name", "ingredients"]
                    }
                },
            )
            # Try to extract assistant message
            content = None
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")

            def escape_dollar(obj):
                if isinstance(obj, str):
                    return obj.replace("$", r"\\$")
                elif isinstance(obj, dict):
                    return {escape_dollar(k): escape_dollar(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [escape_dollar(i) for i in obj]
                else:
                    return obj
            if content:
                st.markdown("## Suggested plan")
                st.write(escape_dollar(content))
            else:
                st.markdown("## Raw response")
                st.write(escape_dollar(resp))
        except Exception as e:
            st.error(f"Failed to generate response: {e}")
else:
    st.info("Fill in the form and click Save Preferences.")

# Product Research Section
st.divider()
st.subheader("Product Research")
st.write("Search for products on e-commerce sites and get AI-powered price research.")

with open('sites.json', 'r') as f:
    sites_config = json.load(f)

sites_dict = {site['name']: site for site in sites_config['sites']}
site_name = st.selectbox("Select a site to search", list(sites_dict.keys()))
product_query = st.text_input("Enter a product to search for", placeholder="e.g., laptop, headphones, book")

if st.button("Search for Product", type="primary"):
    if site_name and product_query:
        site = sites_dict[site_name]
        
        # Create status container for real-time updates
        status_container = st.status(f"AI agent is researching {product_query} on {site_name}...", expanded=True)
        
        try:
            agent = ResearchAgent()
            
            # Stream the agent's progress
            with status_container:
                st.write("🤖 Agent initialized")
                st.write(f"🌐 Target: {site['url']}")
                st.write(f"🔍 Query: {product_query}")
                st.write("---")
                
                # Container for step-by-step updates
                progress_text = st.empty()
                steps_container = st.container()
                
                result = agent.run_with_progress(
                    url=site['url'],
                    search_selector=site['search_bar_selector'],
                    product_query=product_query,
                    progress_callback=lambda msg: steps_container.write(msg)
                )
                
                agent.shutdown()
                progress_text.write("✅ Research complete!")
            
            status_container.update(label="Research complete!", state="complete")
            
            st.markdown("## 🔍 Research Result")
            st.write(result)
            
        except Exception as e:
            status_container.update(label="Error occurred", state="error")
            st.error(f"An error occurred: {e}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
    else:
        st.warning("Please select a site and enter a product query.")


