# budgeat
BudgEat

## Quick start

1. Create and activate a virtual environment (recommended):
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```bash
     python -m venv .venv; .venv\\Scripts\\Activate.ps1
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

The app collects:
- Desired weekly food budget (USD)
- Desired daily calorie intake
- Food preferences (free text)

Results are shown after submitting the form. We'll refine the UI and data model next.
