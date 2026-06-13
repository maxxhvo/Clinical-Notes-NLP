from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import torch

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

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

card_style = {
    "border": "1px solid #ddd",
    "borderRadius": "10px",
    "padding": "20px",
    "marginTop": "20px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
    "backgroundColor": "white"
}

button_style = {
    "fontSize": "18px",
    "padding": "12px 24px",
    "marginTop": "12px",
    "borderRadius": "8px",
    "cursor": "pointer"
}

# Initialize Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX]
)

app.layout = html.Div([
    html.H1([
    "30-Day Readmission Risk Predictor",
    html.Br(),
    "Using MIMIC-IV Medication Reconciliation Notes"
    ]),
    
    dcc.Tabs([
        dcc.Tab(label="Prediction App", children=[
            html.Div([
                html.H2("Enter Medication Reconciliation Text"),

                html.P(
                    "Paste a medication reconciliation section from a discharge summary. "
                    "The app will estimate the patient's 30-day readmission probability "
                    "and classify the patient as higher or lower risk using the selected "
                    f"decision threshold of {threshold:.3f}."
                ),

                dcc.Textarea(
                    id="input-text",
                    placeholder="Paste medication reconciliation section here...",
                    style={
                        "width": "100%",
                        "height": "250px",
                        "fontSize": "16px",
                        "padding": "12px"
                    }
                ),

                html.Button(
                    "Predict Readmission Risk",
                    id="predict-button",
                    style=button_style
                ),
            ], style=card_style),

            html.Div([
                html.H2("Prediction Result"),
                html.Div(id="prediction-output")
            ], style=card_style),
        ]),

        dcc.Tab(label="Example Medication Reconciliation", children=[
            html.H2(
                "Example Medication Reconciliation Section",
                className="mt-4"
            ),
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
            html.H2("Project Overview",
                    className="mt-4"
            ),
            html.P(
                "This proof-of-concept application uses Bio_ClinicalBERT embeddings derived "
                "from medication reconciliation sections of hospital discharge summaries. "
                "These sections summarize a patient's admission medications, discharge "
                "medications, and medication changes during hospitalization. The resulting "
                "text embeddings are used by an XGBoost classifier to estimate 30-day hospital "
                "readmission risk."
            ),

            html.H2("Model Architecture"),

            html.P(
                "The prediction pipeline consists of two stages. First, medication "
                "reconciliation text is converted into dense numerical embeddings using "
                "Bio_ClinicalBERT, a transformer model pretrained on clinical notes. "
                "The medication reconciliation section summarizes a patient's admission "
                "medications, discharge medications, and medication changes made during "
                "hospitalization. The embedding corresponding to the CLS token from the "
                "final transformer layer is used as a fixed-length representation of the "
                "entire note section."
            ),

            html.P(
                "These embeddings are then provided as inputs to an Extreme Gradient "
                "Boosting (XGBoost) classifier. XGBoost is an ensemble learning method "
                "that constructs a sequence of decision trees, where each new tree focuses "
                "on correcting errors made by previous trees. The model outputs a predicted "
                "probability of 30-day hospital readmission, which is subsequently converted "
                "into a high-risk or lower-risk classification using a clinically motivated "
                "decision threshold."
            ),
            html.H3("Embedding Model"),
            html.Ul([
                html.Li(f"Model: {config['embedding_model']}"),
                html.Li("Architecture: Bio_ClinicalBERT transformer"),
                html.Li("Input text: medication reconciliation section"),
                html.Li("Maximum token length: 512"),
                html.Li("Embedding used: CLS token from final hidden layer"),
            ]),

            html.H3("Classifier"),
            html.Ul([
                html.Li("Model type: XGBoost classifier"),
                html.Li("Input features: ClinicalBERT embeddings"),
                html.Li("Output: probability of 30-day readmission"),
                html.Li(f"Decision threshold: {threshold:.3f}"),
            ]),
            html.H3("Model Development"),

            html.P(
                "Several machine learning models and decision thresholds were evaluated "
                "during development. The final model combined Bio_ClinicalBERT embeddings "
                "with an XGBoost classifier because it provided the strongest balance of "
                "discrimination and clinical utility on the held-out test set."
            ),

            html.P(
                "XGBoost hyperparameters were selected through cross-validation, including "
                "parameters controlling tree depth, learning rate, subsampling, and model "
                "complexity. Final model selection emphasized precision-recall performance "
                "and the ability to identify patients at elevated risk of readmission."
            ),

            html.H2("Threshold Selection and Clinical Motivation"),

            html.P(
                "The precision–recall curve shown below summarizes model performance "
                "across all possible classification thresholds."
            ),

            html.Img(
                src="/assets/Precision-Recall Curve.png",
                style={
                    "width": "80%",
                    "maxWidth": "800px",
                    "display": "block",
                    "margin": "auto"
                }
            ),

            html.H3("Threshold Selection Discussion"),

            html.P(
                "For healthcare prediction tasks, classification thresholds should be "
                "selected based on the intended clinical objective rather than solely on "
                "mathematical metrics such as the F1 score. In the context of hospital "
                "readmission prediction, missing a patient who will ultimately be readmitted "
                "may be more consequential than incorrectly flagging a patient who will not "
                "be readmitted, since many follow-up interventions are relatively low cost."
            ),

            html.P(
                "To support this objective, the operating threshold was selected using the "
                "precision–recall curve rather than maximizing the F1 score. A threshold of "
                "0.212 achieved approximately 70% recall and 29% precision on the test set. "
                "This operating point identifies the majority of future readmissions while "
                "still enriching the population for higher-risk patients, making it a more "
                "practical threshold for targeted post-discharge interventions such as "
                "follow-up calls, medication reconciliation, and care coordination."
            ),

            html.H2("Author and Sources"),
            html.P("Author: Max Vo  | GitHub: maxxhvo"),

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
],
style={
    "maxWidth": "1100px",
    "margin": "0 auto",
    "padding": "30px"
}
)

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
        html.Div(
            "Higher risk" if high_risk else "Lower risk",
            style={
                "fontSize": "26px",
                "fontWeight": "bold",
                "color": "crimson" if high_risk else "green",
                "marginBottom": "12px"
            }
        ),

        html.P([
            html.Strong("Predicted readmission probability: "),
            f"{prob:.3f}"
        ]),

        html.P([
            html.Strong("Classification threshold: "),
            f"{threshold:.3f}"
        ]),

        html.P(
            "Patients with predicted probabilities greater than or equal to the "
            "threshold are classified as higher risk."
        )
    ])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)