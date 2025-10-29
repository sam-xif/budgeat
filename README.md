# BudgEat 🥗

AI-powered grocery price research with vision capabilities. Uses NVIDIA's Nemotron VLM to search multiple stores and find the best prices.

## Features

- 🤖 **Agentic AI** - Uses LangGraph with NVIDIA Nemotron for intelligent decision-making
- 👁️ **Vision AI** - Can literally read page screenshots when JavaScript renders content
- 🏪 **Multi-store** - Searches Target, Amazon, Walmart, Kroger automatically
- ⚡ **Fast** - Optimized with headless browsing and smart waits
- 📊 **Batch Processing** - Research entire recipes automatically

3. Create a .env file with your NVIDIA API key:
   ```bash
   echo "NVIDIA_API_KEY=your_key_here" > .env
   ```

4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

### 1. Setup Environment

Results are shown after submitting the form. We'll refine the UI and data model next.



## High level idea
An AI agent that shops for you according to your budgetary, caloric intake, and cuisine preferences.
This takes the burden off of, say, parents who may be operating on a tight budget and don't have the time to 
enumerate all of the food options available. 



## Workflow
1. Ingest user preferences
2. Call spoonacular to get a set of recipes that can be cooked based on preference items (lookup first 10 recipes for simplicity)
3. Produce a mapping from recipe to list of ingredients
4. Perform a greedy algorithm to find the set of recipes for which common forms of the ingredients are within budget
5. Return the list of recipes to the user and produce a shopping list to take with them to the store

Nice-to-haves later: 
1. actually compute prices based on amounts of ingredients. 
2. actually place the order on instacart or something


