import re
from typing import Dict, List, Set

from schemas import Product, ProductPair, SupportEvidence


STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "with", "for", "to", "of", "in",
    "on", "at", "by", "is", "are", "this", "that", "it", "its",
    "from", "as", "be", "can", "will", "into", "your"
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokens(text: str) -> List[str]:
    text = normalize_text(text)
    parts = text.split()

    out: List[str] = []
    for token in parts:
        if token not in STOP_WORDS and len(token) >= 3:
            out.append(token)
    return out


def product_text(product: Product) -> str:
    chunks: List[str] = []
    chunks.append(product.title)
    chunks.append(product.description)
    chunks.append(product.category)

    for summary in product.image_summaries:
        chunks.append(summary)

    for key in ["brand", "style", "color", "material", "room", "price"]:
        if key in product.metadata:
            chunks.append(f"{key}: {product.metadata[key]}")

    return normalize_text(" ".join(str(x) for x in chunks if x))


def pair_text(pair: ProductPair) -> str:
    left = product_text(pair.product_a)
    right = product_text(pair.product_b)
    return f"product a {left} product b {right}"


class FeatureBuilder:
    """
    Deterministic support layer.
    This does NOT make the final compatibility decision.
    It only gives the LLM structured context and weak signals.
    """

    def build(self, pair: ProductPair) -> SupportEvidence:
        text_a = product_text(pair.product_a)
        text_b = product_text(pair.product_b)

        tokens_a = set(simple_tokens(text_a))
        tokens_b = set(simple_tokens(text_b))

        shared = list(tokens_a & tokens_b)
        shared.sort()

        union = tokens_a | tokens_b
        if len(union) == 0:
            token_jaccard = 0.0
        else:
            token_jaccard = len(tokens_a & tokens_b) / len(union)

        notes: List[str] = []
        if not pair.product_a.image_refs and not pair.product_a.image_summaries:
            notes.append("Product A has no image input.")
        if not pair.product_b.image_refs and not pair.product_b.image_summaries:
            notes.append("Product B has no image input.")

        price_a = pair.product_a.metadata.get("price")
        price_b = pair.product_b.metadata.get("price")
        price_ratio = None
        if isinstance(price_a, (int, float)) and isinstance(price_b, (int, float)) and price_b != 0:
            hi = max(price_a, price_b)
            lo = min(price_a, price_b)
            price_ratio = hi / lo

        structured_context: Dict[str, object] = {
            "category_a": pair.product_a.category,
            "category_b": pair.product_b.category,
            "brand_a": pair.product_a.metadata.get("brand"),
            "brand_b": pair.product_b.metadata.get("brand"),
            "price_a": price_a,
            "price_b": price_b,
            "price_ratio": price_ratio,
            "image_count_a": len(pair.product_a.image_refs),
            "image_count_b": len(pair.product_b.image_refs),
            "has_image_summaries_a": len(pair.product_a.image_summaries) > 0,
            "has_image_summaries_b": len(pair.product_b.image_summaries) > 0,
        }

        soft_signals: Dict[str, object] = {
            "shared_keywords": shared[:15],
            "token_jaccard": round(token_jaccard, 4),
            "same_category": normalize_text(pair.product_a.category) == normalize_text(pair.product_b.category),
            "title_a": pair.product_a.title,
            "title_b": pair.product_b.title,
            "image_summaries_a": pair.product_a.image_summaries,
            "image_summaries_b": pair.product_b.image_summaries,
        }

        return SupportEvidence(
            structured_context=structured_context,
            soft_signals=soft_signals,
            notes=notes,
        )