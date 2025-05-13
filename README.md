# Translation Evaluation via Question Answering (TREQA)

The `treqa` package contains code for *extrinsic* evaluation of machine translation through question-answering, using LLMs for both generating and answering questions.

This is the official repository for the paper ["Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering"](https://arxiv.org/abs/2504.07583).

<hr />

> **Abstract:** *Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at this https URL*
<hr />

## Installation

To install the `treqa` package, run:

```bash
pip install -e .
```

Additionally, some of `treqa`'s features require additional dependencies.

```bash
pip install lmqg # for lmqg-based question generation
pip install unbabel-comet # for comet-based scoring
pip install git+https://github.com/sweta20/metricx # for metricx-based scoring
```

## Usage

### Generating Question-Answer Pairs

For most cases, you will first need to generate question-answer pairs based on the source text and/or reference translation.
To do this, run:

```bash
treqa-generate qa_file.jsonl --src data/example.src.pt --tgt data/example.ref.en
```

This will instantiate and prompt an LLM ([Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by default) to generate question-answer pairs for the given source and reference translations.

If you already know which candidate translations you want to evaluate, you can pass them to generation script as well, which should lead to more *targeted* question-answer pairs.

```bash
treqa-generate qa_file.jsonl \
    --src data/example.src.pt \
    --tgt data/example.ref.en \
    --hyp data/example.cands.1.en data/example.cands.2.en
```

### Evaluating Translations

After generating question-answer pairs, you can evaluate a candidate translation using TREQA by running:

```bash
treqa-evaluate \
    --hyp data/example.cands.1.en \
    --src data/example.src.pt \
    --ref data/example.ref.en \
    --qa-file qa_file.jsonl \
    --save-detailed-evaluation per-hyp-detailed.1.jsonl 
```

The last argument is optional and will save the detailed evaluation results for each hypothesis, including the predicted answers for each question when using the translation as the passage aswell as the original (or recomputed) answers using the source/reference as the passage, and the answer matching scores.


### Using LLM APIs

You can also query LLM APIs (including closed LLM providers or your own self-hosted ones) through [LiteLLM](https://github.com/BerriAI/litellm) by setting the right arguments.

```bash
export LLM_ARGS="{\"provider\": \"litellm\", \"model_name\": \"litellm_proxy/your_proxied_model\", \"api_key\": \"your_api_key\"}"
treqa-generate qa_file.jsonl \
    --src data/example.src.pt \
    --tgt data/example.ref.en \
    --hyp data/example.cands.1.en data/example.cands.2.en \
    --qg-model-args "$LLM_ARGS"
treqa-evaluate \
    --hyp data/example.cands.1.en \
    --src data/example.src.pt \
    --ref data/example.ref.en \
    --qa-file qa_file.jsonl \
    --qa-model-args "$LLM_ARGS" \
    --save-detailed-evaluation per-hyp-detailed.1.jsonl \
```