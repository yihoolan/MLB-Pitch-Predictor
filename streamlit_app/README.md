# streamlit_app

Streamlit dashboard that calls the FastAPI service at `http://localhost:8001` to display pitch-type predictions.

---

| File | Description |
| --- | --- |
| `app.py` | Streamlit dashboard that lets users search for a pitcher and batter, set the game situation, and POST to the FastAPI `/predict` endpoint to display a pitch-type probability chart; includes a What-If explorer for comparing predictions across different counts. |
