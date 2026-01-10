from transformers import pipeline

# External research AI (fallback mode)
external_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512
)

def external_research_answer(question):
    prompt = f"""
You are a clinical research assistant.
Answer using general medical research knowledge.

Question:
{question}

Provide professional medical explanation.
"""

    result = external_llm(prompt)[0]["generated_text"]

    return {
        "answer": result,
        "source": "External Medical Research (AI)"
    }
