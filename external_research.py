from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def external_research_answer(query):
    response = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[
            {"role": "system", "content": "You are a medical research AI. Provide evidence-based medical information."},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return {
        "answer": response.choices[0].message.content
    }
