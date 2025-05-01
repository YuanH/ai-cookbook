from openai import OpenAI

client = OpenAI(api_key="api-key")


def generate(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""Use the context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content