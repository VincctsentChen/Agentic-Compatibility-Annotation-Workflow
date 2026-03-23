import re
import uuid
from typing import Dict, List

from schemas import LearnedRule, PolicyLearningResult, PolicySlice, RepresentativeExample


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class PolicyMemory:
    def __init__(
        self,
        max_rules_in_memory: int = 100,
        max_examples_in_memory: int = 30,
    ) -> None:
        self.max_rules_in_memory = max_rules_in_memory
        self.max_examples_in_memory = max_examples_in_memory

        self.rules: Dict[str, LearnedRule] = {}
        self.examples: Dict[str, RepresentativeExample] = {}

    def add_learning(
        self,
        learning: PolicyLearningResult,
        pair_id: str,
        true_label: str,
    ) -> None:
        for rule_text in learning.rules:
            self._add_rule(rule_text, pair_id)

        if learning.store_example:
            self._add_example(
                compact_summary=learning.example_summary,
                lesson=learning.example_lesson,
                true_label=true_label,
                pair_id=pair_id,
            )

        self._prune()

    def _add_rule(self, rule_text: str, pair_id: str) -> None:
        key = _normalize(rule_text)
        if not key:
            return

        if key in self.rules:
            self.rules[key].support += 1
            self.rules[key].weight += 1.0
            if pair_id not in self.rules[key].source_pair_ids:
                self.rules[key].source_pair_ids.append(pair_id)
            return

        self.rules[key] = LearnedRule(
            rule_id=str(uuid.uuid4())[:8],
            text=rule_text.strip(),
            support=1,
            weight=1.0,
            source_pair_ids=[pair_id],
        )

    def _add_example(
        self,
        compact_summary: str,
        lesson: str,
        true_label: str,
        pair_id: str,
    ) -> None:
        key = _normalize(lesson)
        if not key:
            return

        if key in self.examples:
            self.examples[key].support += 1
            self.examples[key].weight += 1.0
            if pair_id not in self.examples[key].source_pair_ids:
                self.examples[key].source_pair_ids.append(pair_id)
            return

        self.examples[key] = RepresentativeExample(
            example_id=str(uuid.uuid4())[:8],
            compact_summary=compact_summary.strip(),
            lesson=lesson.strip(),
            true_label=true_label,
            support=1,
            weight=1.0,
            source_pair_ids=[pair_id],
        )

    def _prune(self) -> None:
        if len(self.rules) > self.max_rules_in_memory:
            items = list(self.rules.items())
            items.sort(key=lambda x: (x[1].weight, x[1].support), reverse=True)
            items = items[: self.max_rules_in_memory]
            self.rules = dict(items)

        if len(self.examples) > self.max_examples_in_memory:
            items = list(self.examples.items())
            items.sort(key=lambda x: (x[1].weight, x[1].support), reverse=True)
            items = items[: self.max_examples_in_memory]
            self.examples = dict(items)

    def build_policy_slice(
        self,
        max_rules: int = 6,
        max_examples: int = 2,
        char_budget: int = 900,
    ) -> PolicySlice:
        """
        Keep prompt short:
        - prioritize rules over examples
        - keep examples only if room remains
        """
        selected_rules: List[str] = []
        selected_examples: List[Dict[str, str]] = []

        remaining = char_budget

        rule_items = list(self.rules.values())
        rule_items.sort(key=lambda x: (x.weight, x.support), reverse=True)

        for rule in rule_items:
            text = f"- {rule.text}"
            cost = len(text)
            if len(selected_rules) >= max_rules:
                break
            if cost <= remaining:
                selected_rules.append(rule.text)
                remaining -= cost

        example_items = list(self.examples.values())
        example_items.sort(key=lambda x: (x.weight, x.support), reverse=True)

        for ex in example_items:
            if len(selected_examples) >= max_examples:
                break

            compact = {
                "lesson": ex.lesson,
                "summary": ex.compact_summary,
                "true_label": ex.true_label,
            }

            cost = len(ex.lesson) + len(ex.compact_summary) + len(ex.true_label) + 30
            if cost <= remaining:
                selected_examples.append(compact)
                remaining -= cost

        return PolicySlice(
            rules=selected_rules,
            examples=selected_examples,
        )

    def debug_dump(self) -> None:
        print("\nLEARNED RULES")
        for rule in sorted(self.rules.values(), key=lambda x: (x.weight, x.support), reverse=True):
            print(f"  [{rule.support}] {rule.text}")

        print("\nREPRESENTATIVE EXAMPLES")
        for ex in sorted(self.examples.values(), key=lambda x: (x.weight, x.support), reverse=True):
            print(f"  [{ex.support}] lesson={ex.lesson}")
            print(f"      summary={ex.compact_summary}")