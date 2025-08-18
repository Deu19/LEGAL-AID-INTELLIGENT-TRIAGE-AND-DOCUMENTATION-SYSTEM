# app_flask.py

from flask import Flask, render_template_string, request
from legal_aid_system import NLPProcessor, TriageModule, DocumentGenerator, RecommendationEngine, SecurityLayer
import pathlib

import csv

# Initialize backend components
nlp = NLPProcessor()
triage = TriageModule()
doc_gen = DocumentGenerator()
rec_engine = RecommendationEngine()
security = SecurityLayer()

# Load training data from CSV and train model
texts, labels = [], []
with open("legal_triage_dataset.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row["case_text"])
        labels.append(triage.outcome_labels.index(row["outcome_label"]))
triage.train(texts, labels)

app = Flask(__name__)

# -----------------------------------------------------------------------
# HTML templates
# -----------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Legal Aid Demo</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 30px auto; }
    textarea { width: 100%; min-height: 140px; }
    .result { margin-top: 20px; padding: 15px; background:#f0f4ff; border-radius: 8px; }
    button { padding:8px 14px; border-radius:6px; }
  </style>
</head>
<body>
  <h1>AI Legal Aid – Demo</h1>
  <form method="POST" action="/">
    <textarea name="case_text" placeholder="Describe your legal issue here...">{{ case_text or "" }}</textarea><br><br>
    <button type="submit">Analyze</button>
  </form>

  {% if result %}
    <div class="result">
      <strong>Plain Summary:</strong> {{ result.summary }}<br>
      <strong>Predicted Outcome:</strong> {{ result.pred_outcome }} ({{ result.win_prob }} chance)<br>
      <strong>Complexity:</strong> {{ result.complexity }}<br>
      <strong>Pro Bono Suggestion:</strong> {{ result.pro_bono }}
      <br><br>
      <form method="POST" action="/affidavit">
        <input type="hidden" name="raw_text" value="{{ result.raw }}">
        <button type="submit">Generate Affidavit</button>
      </form>
    </div>
  {% endif %}
</body>
</html>
"""

AFFIDAVIT_FORM_HTML = """
<!DOCTYPE html>
<html>
<head><title>Generate Affidavit</title></head>
<body>
  <h1>Generate Affidavit</h1>
  <form method="POST" action="/generate">
    <input type="hidden" name="raw_text" value="{{ raw_text }}">
    Full Name: <input type="text" name="full_name" required><br><br>
    <button type="submit">Create Affidavit</button>
  </form>
</body>
</html>
"""

GENERATED_HTML = """
<!DOCTYPE html>
<html>
<head><title>Affidavit Generated</title></head>
<body>
  <h1>Affidavit Generated ✅</h1>
  <p>File saved as: <strong>{{ path }}</strong></p>
  <a href="/">Go back</a>
</body>
</html>
"""

# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    case_text = ""
    if request.method == "POST":
        case_text = request.form["case_text"]
        if case_text:
            summary = nlp.preprocess(case_text)
            tri = triage.predict_outcome(summary)
            rec = rec_engine.match("family", "NY")[0]["name"]
            result = {
                "raw": case_text,
                "summary": summary,
                "pred_outcome": tri.predicted_outcome,
                "win_prob": f"{tri.win_probability:.0%}",
                "complexity": tri.complexity,
                "pro_bono": rec,
            }
    return render_template_string(INDEX_HTML, result=result, case_text=case_text)


@app.route("/affidavit", methods=["POST"])
def affidavit_form():
    raw_text = request.form["raw_text"]
    return render_template_string(AFFIDAVIT_FORM_HTML, raw_text=raw_text)


@app.route("/generate", methods=["POST"])
def generate_affidavit():
    raw_text = request.form["raw_text"]
    full_name = request.form["full_name"]
    # Generate document and rename the file using <full_name>
    base_path = doc_gen.generate_form("generic_affidavit", {"client": full_name, "body": raw_text})
    final_path = f"affidavit_{full_name}.txt"
    pathlib.Path(base_path).rename(final_path)
    return render_template_string(GENERATED_HTML, path=final_path)


if __name__ == "__main__":
    app.run(debug=True)