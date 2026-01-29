import art
import openai
import random
import regex
from pydantic import BaseModel
import time
import weave

from summarizer.get_judge_completion import get_judge_completion
from summarizer.load_documents import Document


class SummarizerScenario(BaseModel):
    doc: Document
    step: int = 0
    use_full: bool = False


@weave.op()
@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(model: art.Model, scenario: SummarizerScenario) -> art.Trajectory:
    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": """
You are a specialized AI assistant that generates concise, informative summaries for documents.
""",
            }
        ],
        reward=0,
        metrics={
            "word_count": 0,
            "len": 0,
            "percent": 0,
            "percent_full": 0,
            "percent_diff": 0,
        },
    )

    summarize_prompt = f"""You are a specialized AI assistant that generates concise, informative summaries for documents.

Here is a document: {scenario.doc.document_text}

Generate a summary that conveys all relevant information in a concise manner."""

    trajectory.messages_and_choices.append(
        {"role": "user", "content": summarize_prompt}
    )

    requested_at = int(time.time() * 1000)

    messages = trajectory.messages()
    completion = await client.chat.completions.create(
        model=model.inference_model_name, messages=messages, max_tokens=1000
    )
    choice = completion.choices[0]
    if scenario.use_full:
        choice.message.content = scenario.doc.document_text
    trajectory.messages_and_choices.append(choice)
    summary = choice.message.content

    total_score = 0
    total_score_full = 0
    total_questions = 0

    for question in scenario.doc.questions:
        total_questions += 1
        if not regex.search(r"\p{Han}", summary) and len(summary) <= 3000:
            prompt = f"Here is a document: {summary}\n\nAnswer this question to the best of your ability in one sentence, if the document does not contain the answer, just state so: {question.q}"
            response = await get_judge_completion(prompt)

            judge_prompt = f"Here is a document: {scenario.doc.document_text}\n\nHere is a question: {question.q}\n\nHere is a generated answer: {response}\n\nHere is the golden answer: {question.a}\n\nIf the answers mostly match return a 1, if they do not match return a 0. Do not return any other text."

            score = await get_judge_completion(judge_prompt)
            try:
                total_score += int(score)
            except:
                pass

        prompt_full = f"Here is a document: {scenario.doc.document_text}\n\nAnswer this question to the best of your ability in one sentence, if the document does not contain the answer, just state so: {question.q}"
        response_full = await get_judge_completion(prompt_full)

        judge_prompt_full = f"Here is a document: {scenario.doc.document_text}\n\nHere is a question: {question.q}\n\nHere is a generated answer: {response_full}\n\nHere is the golden answer: {question.a}\n\nIf the answers mostly match return a 1, if they do not match return a 0. Do not return any other text."

        score_full = await get_judge_completion(judge_prompt_full)
        try:
            total_score_full += int(score_full)
        except:
            pass

        if not regex.search(r"\p{Han}", summary) and len(summary) <= 3000:
            if random.random() < 0.05:
                print("Answers:")
                print(summary)
                print("question", question.q)
                print("golden:", question.a)
                print("generated:", response)
                print("score:", score)
                print("score-full:", score_full)
                print("generated-full:", response_full)
                print("\n\n\n\n\n")

    trajectory.metrics["percent"] = total_score / total_questions
    trajectory.metrics["percent_full"] = total_score_full / total_questions
    trajectory.metrics["percent_diff"] = (
        trajectory.metrics["percent"] - trajectory.metrics["percent_full"]
    )
    trajectory.metrics["word_count"] = len(summary.split())
    trajectory.metrics["len"] = len(summary)
    trajectory.reward = total_score

    return trajectory
