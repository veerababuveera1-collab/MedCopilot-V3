import os
from groq import Groq

def external_research_answer(query: str):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a clinical research assistant. Provide evidence-based medical answers."},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=700
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "source": "Groq LLaMA-3.1"
    }
