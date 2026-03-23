from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import (
    ProductPair,
    RetrievedExample,
    consensus_strength,
    label_distribution,
    majority_label,
)
from tools import pair_text


class ExampleMemory:
    """
    Memory built only from human-labeled training pairs.
    Used to retrieve similar examples for few-shot guidance.
    """

    def __init__(self) -> None:
        self.records: List[ProductPair] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = None

    def fit(self, records: List[ProductPair]) -> None:
        labeled_records: List[ProductPair] = []
        corpus: List[str] = []

        for record in records:
            if record.human_labels:
                labeled_records.append(record)
                corpus.append(pair_text(record))

        self.records = labeled_records

        if not corpus:
            self.matrix = None
            return

        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_pair: ProductPair, k: int = 4) -> List[RetrievedExample]:
        if not self.records or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([pair_text(query_pair)])
        sims = cosine_similarity(query_vec, self.matrix)[0]

        indices = list(range(len(sims)))
        indices.sort(key=lambda i: sims[i], reverse=True)

        out: List[RetrievedExample] = []
        for idx in indices[:k]:
            record = self.records[idx]

            pair_summary = (
                f"A: {record.product_a.title} [{record.product_a.category}] | "
                f"B: {record.product_b.title} [{record.product_b.category}]"
            )

            out.append(
            RetrievedExample(
                pair_id=record.pair_id,
                similarity=float(sims[idx]),
                consensus_label=majority_label(record.human_labels),
                human_distribution=label_distribution(record.human_labels),
                consensus_strength=consensus_strength(record.human_labels),
                pair_summary=pair_summary,
                short_text="",
    )
)
        return out