from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import torch

from dash import Dash, html, dcc, Input, Output, State
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "App_Artifacts"

model = joblib.load(ARTIFACT_DIR / "xgb_readmission_meds.joblib")
examples = pd.read_parquet(ARTIFACT_DIR / "medication_examples.parquet")

with open(ARTIFACT_DIR / "model_config.json") as f:
    config = json.load(f)

threshold = config["threshold"]

tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
bert = AutoModel.from_pretrained(config["embedding_model"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = bert.to(device)
bert.eval()


def embed_text(text):
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = bert(**tokens)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding


app = Dash(__name__)

app.layout = html.Div([
    html.H1("30-Day Readmission Risk Predictor"),

    html.P(
        "This proof-of-concept app uses Bio_ClinicalBERT embeddings from the "
        "medication reconciliation section of discharge summaries and an XGBoost "
        "classifier to estimate 30-day readmission risk."
    ),

    html.H2("Example Medication Reconciliation Section"),

    dcc.Dropdown(
        id="example-dropdown",
        options=[
            {
                "label": f"Example {i+1} | Readmit: {row.readmit_30d}",
                "value": i
            }
            for i, row in examples.reset_index(drop=True).iterrows()
        ],
        value=0
    ),

    html.Pre(
        id="example-text",
        style={
            "whiteSpace": "pre-wrap",
            "border": "1px solid #ddd",
            "padding": "1rem",
            "maxHeight": "300px",
            "overflowY": "scroll"
        }
    ),

    html.H2("Enter Medication Reconciliation Text"),

    dcc.Textarea(
        id="input-text",
        placeholder="Paste medication reconciliation section here...",
        style={"width": "100%", "height": "250px"}
    ),

    html.Button("Predict Readmission Risk", id="predict-button"),

    html.H2("Prediction"),

    html.Div(id="prediction-output"),

    html.H2("Threshold Explanation"),

    html.P(
        f"The model uses a threshold of {threshold:.3f}, selected to target "
        f"approximately {config['target_recall']:.0%} recall. This prioritizes "
        "identifying most future readmissions while accepting some false positives."
    )
])


@app.callback(
    Output("example-text", "children"),
    Input("example-dropdown", "value")
)
def show_example(example_idx):
    row = examples.iloc[example_idx]
    return row["medication_reconciliation"]


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("input-text", "value")
)
def predict_readmission(n_clicks, text):
    if not n_clicks:
        return "Enter text and click predict."

    if text is None or text.strip() == "":
        return "Please enter medication reconciliation text."

    embedding = embed_text(text)
    prob = model.predict_proba(embedding)[:, 1][0]
    high_risk = prob >= threshold

    return html.Div([
        html.H3(f"Predicted readmission probability: {prob:.3f}"),
        html.H3(
            "High risk" if high_risk else "Lower risk",
            style={"color": "crimson" if high_risk else "green"}
        ),
        html.P(f"Decision threshold: {threshold:.3f}")
    ])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)