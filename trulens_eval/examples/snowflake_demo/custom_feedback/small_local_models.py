import os
from typing import Tuple

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from scipy.special import expit
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from trulens_eval import Provider
from trulens_eval.feedback import prompts

CONTEXT_RELEVANCE_MODEL_PATH = os.getenv(
    "SMALL_LOCAL_MODELS_CONTEXT_RELEVANCE_MODEL_PATH",
    "/trulens_demo/small_local_models/context_relevance",
)
GROUNDEDNESS_MODEL_PATH = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"


class SmallLocalModels(Provider):

    # Context relevance.
    context_relevance_tokenizer: PreTrainedTokenizerBase = (
        AutoTokenizer.from_pretrained(CONTEXT_RELEVANCE_MODEL_PATH)
    )
    context_relevance_model: PreTrainedModel = (
        AutoModelForSequenceClassification.
        from_pretrained(CONTEXT_RELEVANCE_MODEL_PATH)
    )

    # Groundedness.
    groundedness_tokenizer: PreTrainedTokenizerBase = (
        AutoTokenizer.from_pretrained(GROUNDEDNESS_MODEL_PATH, use_fast=False)
    )
    groundedness_model: PreTrainedModel = (
        AutoModelForSequenceClassification.
        from_pretrained(GROUNDEDNESS_MODEL_PATH)
    )

    def context_relevance(
        self, question: str, context: str, temperature: float = 0.0
    ) -> float:
        tokenizer = self.context_relevance_tokenizer
        model = self.context_relevance_model
        with torch.no_grad():
            logit = model.forward(
                torch.tensor(
                    tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(f"{question} [SEP] {context}")
                    )
                ).reshape(1, -1)
            ).logits.numpy()
            if logit.size != 1:
                raise ValueError("Unexpected number of results from model!")
            logit = float(logit[0, 0])
            return expit(logit)

    def groundedness_measure_with_nli(self, source: str,
                                      statement: str) -> Tuple[float, dict]:
        nltk.download('punkt', quiet=True)
        groundedness_scores = {}

        reasons_str = ""
        if isinstance(source, list):
            source = ' '.join(map(str, source))
        hypotheses = sent_tokenize(statement)
        for i, hypothesis in enumerate(tqdm(
                hypotheses, desc="Groundendess per statement in source")):
            score = self._doc_groundedness(
                premise=source, hypothesis=hypothesis
            )
            reasons_str = reasons_str + str.format(
                prompts.GROUNDEDNESS_REASON_TEMPLATE,
                statement_sentence=hypothesis,
                supporting_evidence="[Doc NLI Used full source]",
                score=score * 10,
            )
            groundedness_scores[f"statement_{i}"] = score
        average_groundedness_score = float(
            np.mean(list(groundedness_scores.values()))
        )
        return average_groundedness_score, {"reasons": reasons_str}

    def _doc_groundedness(self, premise: str, hypothesis: str) -> float:
        tokenizer = self.groundedness_tokenizer
        model = self.groundedness_model
        with torch.no_grad():
            tokens = tokenizer(
                premise, hypothesis, truncation=True, return_tensors="pt"
            )
            output = model(tokens["input_ids"])
            prediction = torch.softmax(output["logits"][0], -1).tolist()
            return prediction[0]
