import re
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class SpacyTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = nlp.Defaults.stop_words

    def clean_text(self, text):
        text = text.lower()

        text = re.sub(r"(http|https|ftp|ssh)://\S+", "", text)

        text = re.sub(r"<.*?>", "", text)

        text = re.sub(r"[^a-z0-9\s-]", "", text)

        doc = nlp(text)
        text = " ".join(
            [token.lemma_ for token in doc if token.text not in self.stop_words]
        )

        return text.strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]
