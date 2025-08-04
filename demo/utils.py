SYSTEM_PROMPT_QAG = "You are a helpful AI assistant skilled in generating questions and answers from given passages."

QAG_TEMPLATE = """"Generate question-answer pairs to verify translation accuracy. Each answer should be a key phrase, concept, or entity from the original passage (source or reference) that could help detect errors or mistranslations in the candidate(s).
The questions and answers must be strictly in English, while ensuring that the meaning of the answer is preserved. The questions should be diverse and cover different aspects of the passage. Answer in the format:

Q: <question1>
A: <answer1>

Q: <question2>
A: <answer2>

...

Source Passage:
{src_passage}

Reference Passage:
{ref_passage}

Candidate Passage(s):
{alternatives}

Question-Answer Pairs:
"""

SYSTEM_PROMPT_QA = (
    """You are a helpful AI assistant skilled in question answering."""
)

QA_TEMPLATE = """Given the following passage and question, return the answer in English using only the information from the passage. The answer should be a concise response based on the provided content.
###
Passage:
{passage}
###
Question:
{question}
###
Answer:"""

def parse_output_default(
        output: str,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    pairs = []
    skipped_outs = []
    for qa in output.split("\n\n"):
        # skip if there is no content
        if not qa.strip():
            reason = "no content"
            skipped_outs.append((reason, qa))
            continue
        # skip if there are not two lines
        if len(qa.split("\n")) != 2:
            reason = "not two lines"
            skipped_outs.append((reason, qa))
            continue

        q, a = qa.split("\n")
        # skip if the qa pair don't start with Q: and A:
        if not q.startswith("Q:") or not a.startswith("A:"):
            reason = "no Q: or A:"
            skipped_outs.append((reason, qa))
            continue

        q = q.replace("Q:", "").strip()
        a = a.replace("A:", "").strip()
        pairs.append((q, a))

    return list(set(pairs))