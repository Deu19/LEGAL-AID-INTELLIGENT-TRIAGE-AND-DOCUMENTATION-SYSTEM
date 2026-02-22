from _future_ import annotations

import csv
import json
import pathlib
import datetime as _dt
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import torch
import spacy
from tabulate import tabulate
from cryptography.fernet import Fernet
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


# ---------------------------------------------------------------------------
# SecurityLayer – simple wrapper around Fernet symmetric encryption
# ---------------------------------------------------------------------------
class SecurityLayer:
    def _init_(self, key: Optional[bytes] = None):
        self._key = key or Fernet.generate_key()
        self._fernet = Fernet(self._key)

    @property
    def key(self) -> bytes:
        return self._key

    def encrypt(self, message: str) -> str:
        return self._fernet.encrypt(message.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode()).decode()


# ---------------------------------------------------------------------------
# NLPProcessor – handles basic cleaning + embeddings
# ---------------------------------------------------------------------------
class NLPProcessor:
    def _init_(self):
        self._spacy = spacy.load("en_core_web_sm")
        self._sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def preprocess(self, text: str) -> str:
        doc = self._spacy(text.lower())
        tokens = [
            t.lemma_
            for t in doc
            if not (t.is_punct or t.is_space or t.is_stop)
        ]
        return " ".join(tokens)

    def sentiment(self, text: str) -> Dict[str, Any]:
        return self._sentiment(text)[0]


# ---------------------------------------------------------------------------
# DocumentGenerator – naive template filler
# ---------------------------------------------------------------------------
class DocumentGenerator:
    def _init_(self, template_dir: str = "templates"):
        self.templates_path = pathlib.Path(template_dir)
        self.templates_path.mkdir(exist_ok=True)

    def generate_form(self, template_name: str, user_data: Dict[str, Any]) -> str:
        template_file = self.templates_path / f"{template_name}.json"
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template_name}' not found in {self.templates_path}"
            )
        with open(template_file, "r", encoding="utf-8") as fp:
            template = json.load(fp)["body"]
        document = template.format(**user_data)
        out_path = (
            self.templates_path
            / f"output_{template_name}_{_dt.datetime.now().date()}.txt"
        )
        out_path.write_text(document, encoding="utf-8")
        return str(out_path)


# ---------------------------------------------------------------------------
# TriageModule – tiny ML model (Sentence Embeddings + log-reg)
# ---------------------------------------------------------------------------
@dataclass
class TriagePrediction:
    complexity: str
    win_probability: float
    predicted_outcome: str
    explanation: str


class TriageModule:
    def _init_(self, model_path: Union[str, pathlib.Path, None] = None):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = LogisticRegression(max_iter=1000)
        self.outcome_labels = [
            "Dismissed",
            "Settled",
            "Won",
            "Needs more evidence",
        ]
        if model_path and pathlib.Path(model_path).exists():
            self.load(model_path)

    def train(self, texts: List[str], labels: List[int]) -> None:
        X = self.encoder.encode(texts, convert_to_tensor=False)
        self.model.fit(X, labels)
        preds = self.model.predict(X)
        print(
            "[TriageModule] Training report:\n",
            classification_report(labels, preds, target_names=self.outcome_labels),
        )

    def predict_outcome(self, case_text: str) -> TriagePrediction:
        X = self.encoder.encode([case_text], convert_to_tensor=False)

        probabilities = self.model.predict_proba(X)[0]
        best_idx = probabilities.argmax()

        predicted_label = self.outcome_labels[best_idx]
        confidence = probabilities[best_idx]

        complexity = self._estimate_complexity(case_text)

        # ---------------------------
        # Explainable AI logic
        # ---------------------------
        explanation = self._generate_explanation(X[0], best_idx)

        return TriagePrediction(
            complexity=complexity,
            win_probability=confidence,
            predicted_outcome=predicted_label,
            explanation=explanation,
        )
    
    def _generate_explanation(self, input_vector, class_index):
   
        weights = self.model.coef_[class_index]

        # Contribution = weight * feature value
        contributions = weights * input_vector

        # Get top 5 contributing features
        top_indices = contributions.argsort()[-5:]

        explanation_parts = []

        for idx in top_indices:
            weight = weights[idx]
            if weight > 0:
                explanation_parts.append("strong semantic alignment with similar past cases")
            else:
                explanation_parts.append("lack of strong supporting indicators")

        explanation_text = (
            f"The prediction is influenced by key semantic patterns detected in the case description, "
            f"including {', '.join(set(explanation_parts))}."
        )

        return explanation_text

    def _estimate_complexity(self, text: str) -> str:
        word_count = len(text.split())
        if word_count > 300:
            return "High"
        elif word_count > 150:
            return "Medium"
        return "Low"

    def save(self, path: Union[str, pathlib.Path]):
        joblib.dump({"model": self.model}, path)

    def load(self, path: Union[str, pathlib.Path]):
        data = joblib.load(path)
        self.model = data["model"]


# ---------------------------------------------------------------------------
# RecommendationEngine – simple metadata filter + rank
# ---------------------------------------------------------------------------
class RecommendationEngine:
    def _init_(self, directory_path: str | pathlib.Path = "providers.json"):
        self.dir_path = pathlib.Path(directory_path)
        if self.dir_path.exists():
            with open(self.dir_path, "r", encoding="utf-8") as fp:
                self.providers = json.load(fp)
        else:
            self.providers = [
                {
                    "name": "Legal Aid Society",
                    "category": "family",
                    "location": "NY",
                    "contact": "legal-aid.org",
                },
                {
                    "name": "Pro Bono Connect",
                    "category": "housing",
                    "location": "CA",
                    "contact": "probconnect.org",
                },
            ]

    def match(self, case_category: str, user_location: str) -> List[Dict[str, str]]:
        matches = [
            p
            for p in self.providers
            if p["category"] == case_category
        ]
        matches.sort(key=lambda m: m["location"] != user_location)
        return matches[:5]


# ---------------------------------------------------------------------------
# ChatbotInterface – minimal REPL showcasing full flow
# ---------------------------------------------------------------------------
class ChatbotInterface:
    def _init_(
        self,
        nlp: NLPProcessor,
        triage: TriageModule,
        doc_gen: DocumentGenerator,
        recommender: RecommendationEngine,
        security: SecurityLayer,
    ):
        self.nlp = nlp
        self.triage = triage
        self.doc_gen = doc_gen
        self.rec_engine = recommender
        self.security = security

    def _collect_case_details(self) -> Dict[str, str]:
        print("👋 Welcome to the AI Legal Aid Chatbot!\nPlease describe your legal issue in a paragraph:")
        case_text = input("> ")
        plain = self.nlp.preprocess(case_text)
        sentiment = self.nlp.sentiment(case_text)
        return {
            "raw": case_text,
            "clean": plain,
            "sentiment": sentiment["label"],
        }

    def chat(self):
        user_case = self._collect_case_details()

        # triage
        triage_pred = self.triage.predict_outcome(user_case["clean"])
        print("\n— Case Triage —")
        print(f"Complexity: {triage_pred.complexity}")
        print(
            f"Predicted outcome: {triage_pred.predicted_outcome} ({triage_pred.win_probability:.0%} chance)"
        )

        # recommendation
        # NOTE: this is still a placeholder — category detection could also be ML-based
        category_guess = "family"
        recs = self.rec_engine.match(
            category_guess, user_location="NY"
        )
        print("\n— Matching Resources —")
        print(tabulate(recs, headers="keys"))

        # document drafting demo
        want_form = (
            input(
                "\nWould you like to auto-draft a standard template form? (y/n): "
            ).lower()
            == "y"
        )
        if want_form:
            name = input("Your full legal name: ")
            doc_path = self.doc_gen.generate_form(
                "generic_affidavit",
                {"client": name, "body": user_case["raw"]},
            )
            print(f"Draft saved to {doc_path}")

        # encryption sample
        secure_dump = self.security.encrypt(json.dumps(user_case))
        print("\nYour encrypted session token (save carefully):")
        print(secure_dump)

        print("\nThank you for using the AI Legal Aid Chatbot. Goodbye! 👋")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if _name_ == "_main_":
    nlp = NLPProcessor()
    triage = TriageModule()
    doc_gen = DocumentGenerator()
    rec_engine = RecommendationEngine()
    security = SecurityLayer()

    # ---------------------------------------------------------------
    # Load training samples from CSV and train model
    # ---------------------------------------------------------------
    dataset_path = "legal_triage_dataset.csv"

    texts = []
    labels = []

    with open(dataset_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            case_text = row["case_text"]
            label_str = row["outcome_label"]
            label_index = triage.outcome_labels.index(label_str)
            texts.append(case_text)
            labels.append(label_index)

    triage.train(texts, labels)

    # ---------------------------------------------------------------
    # Start Chatbot
    # ---------------------------------------------------------------
    app = ChatbotInterface(nlp, triage, doc_gen, rec_engine, security)
    app.chat()
