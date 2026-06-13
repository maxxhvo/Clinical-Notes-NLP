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
    html.H1("30-Day Readmission Risk Predictor Using Medication Reconciliation Notes"),

    dcc.Tabs([
        dcc.Tab(label="Prediction App", children=[
            html.P(
                "Paste a medication reconciliation section below to generate a predicted "
                "30-day readmission probability and risk classification."
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

            html.H3("Threshold Explanation"),
            html.P(
                f"The model uses a threshold of {threshold:.3f}, selected to target "
                f"approximately {config['target_recall']:.0%} recall. This prioritizes "
                "identifying most future readmissions while accepting some false positives."
            )
        ]),

        dcc.Tab(label="Example Medication Reconciliation", children=[
            html.H2("Example Medication Reconciliation Section"),

            html.P(
                "Select an example medication reconciliation section from the dataset to "
                "preview the kind of clinical text the app expects as input."
            ),

        dcc.Dropdown(
            id="example-dropdown",
            options=[
                {
                    "label": f"Example {i+1} | Readmit: {bool(row.readmit_30d)}",
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
            )
        ]),

        dcc.Tab(label="Project Information", children=[
            html.H2("Project Overview"),
            html.P(
                "This proof-of-concept application uses Bio_ClinicalBERT embeddings derived "
                "from medication reconciliation sections of hospital discharge summaries. "
                "These sections summarize a patient's admission medications, discharge "
                "medications, and medication changes during hospitalization. The resulting "
                "text embeddings are used by an XGBoost classifier to estimate 30-day hospital "
                "readmission risk."
            ),

            html.H3("Embedding Model"),
            html.Ul([
                html.Li(f"Model: {config['embedding_model']}"),
                html.Li("Architecture: ClinicalBERT / BERT-style transformer"),
                html.Li("Input text: medication reconciliation section"),
                html.Li("Maximum token length: 512"),
                html.Li("Embedding used: CLS token from last hidden state"),
            ]),

            html.H3("Classifier"),
            html.Ul([
                html.Li("Model type: XGBoost classifier"),
                html.Li("Input features: ClinicalBERT text embeddings"),
                html.Li("Output: predicted probability of 30-day readmission"),
                html.Li(f"Decision threshold: {threshold:.3f}"),
            ]),

            html.H3("Author and Sources"),
            html.P("Author: Max Vo"),

            html.Ul([
                html.Li([
                    "Study data: ",
                    html.A(
                        "MIMIC-IV-Note v2.2",
                        href="https://physionet.org/content/mimic-iv-note/2.2/",
                        target="_blank"
                    )
                ]),
                html.Li([
                    "Dataset documentation: ",
                    html.A(
                        "MIMIC-IV-Note module documentation",
                        href="https://mimic.mit.edu/docs/IV/modules/note/",
                        target="_blank"
                    )
                ]),
                html.Li([
                    "Code reference: ",
                    html.A(
                        "MIT-LCP mimic-code GitHub repository",
                        href="https://github.com/MIT-LCP/mimic-code",
                        target="_blank"
                    ),
                    " — used the buildmimic script to create a DuckDB database for MIMIC-IV-Note."
                ])
            ])
        ])
    ])
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