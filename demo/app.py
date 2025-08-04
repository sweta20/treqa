from flask import Flask, render_template, request, jsonify
import time
import re
from utils import *
from run_genai import prompt_genai
import sacrebleu

app = Flask(__name__)

def run_treqa(source, reference, candidates):
    """
    A mock function to simulate the TREQA pipeline with server-side logging.
    """
    print("\n[TREQA] Starting evaluation...")

    # --- Step 1: Question Generation (QAG) ---
    print("--- Step 1: Generating Questions (QAG) ---")
    prompt_qag = QAG_TEMPLATE.format(src_passage=source,
                                   ref_passage=reference,
                                   alternatives=candidates)
    response = prompt_genai(SYSTEM_PROMPT_QAG, prompt_qag)

    questions = [x[0] for x in parse_output_default(response)]
    print(f"--- Step 1 COMPLETED. Generated {len(questions)} questions. ---")
    for i, q in enumerate(questions):
        print(f"  - Q{i+1}: {q}")

    # --- Step 2: Question Answering (QA) ---
    print("\n--- Step 2: Answering Questions (QA) ---")
    reference_answers = []
    candidate_answers = [[] for _ in candidates]
    
    for q_idx, question in enumerate(questions):
        print(f"  - Answering question {q_idx + 1}/{len(questions)} for reference and {len(candidates)} candidates...")
        # reference
        prompt_qa_ref = QA_TEMPLATE.format(passage=reference, question=question)
        answer_ref = prompt_genai(SYSTEM_PROMPT_QA, prompt_qa_ref)
        reference_answers.append(answer_ref)
        
        # candidate
        for i, candidate in enumerate(candidates):
            prompt_qa_cand = QA_TEMPLATE.format(passage=candidate, question=question)
            answer_cand = prompt_genai(SYSTEM_PROMPT_QA, prompt_qa_cand)
            candidate_answers[i].append(answer_cand)
    print("--- Step 2 COMPLETED. All questions answered. ---")


    # --- Step 3: Answer Correctness & Scoring ---
    print("\n--- Step 3: Scoring Answers ---")
    scores = []
    
    # Loop over each candidate's list of answers
    for i, cand_answers_list in enumerate(candidate_answers):
        scores_for_candidate = []
        # Loop over each answer, which corresponds to a question (j is the question index)
        for j, cand_ans in enumerate(cand_answers_list):
            # Get the corresponding reference answer for the same question
            ref_ans = reference_answers[j]

            score_obj = sacrebleu.sentence_chrf(cand_ans, [ref_ans])
            score_val = score_obj.score

            threshold = 50.0
            text = "Excellent Match" if score_val > threshold else "Poor Match"
            rating = "high-score" if score_val > threshold else "low-score"

            scores_for_candidate.append({"score": score_val, "text": text, "rating": rating})
        
        scores.append(scores_for_candidate)
    print("--- Step 3 COMPLETED. All answers scored. ---")


    # Simulate network delay for a better user experience
    time.sleep(1)

    print("\n[TREQA] Evaluation finished. Sending results to client.")
    return {
        "questions": questions,
        "reference_answers": reference_answers,
        "candidate_answers": candidate_answers,
        "scores": scores
    }

@app.route('/')
def index():
    """Renders the main home page."""
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """API endpoint to run the TREQA evaluation."""
    data = request.json
    source = data.get('source')
    reference = data.get('reference')
    candidates = data.get('candidates')

    if not source or not reference or not candidates:
        return jsonify({"error": "Missing source, reference or candidates"}), 400

    results = run_treqa(source, reference, candidates)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)