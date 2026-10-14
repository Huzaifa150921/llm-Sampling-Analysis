
import ollama
import time


# prompt = "Explain why the sky appears blue in simple terms."
# prompt = (
#     "A notebook costs $5 and a pen costs $2. "
#     "What is the total cost of 3 notebooks and 4 pens? "
#     "Explain your calculation."
# )

prompt = (
    
    "Return ONLY a valid JSON object with the keys "
    "name, age, and city, using the values Ali, 25, "
    "and Lahore. Do not use Markdown."
)
# model = "qwen2.5:7b"
model = "llama3.1:8b"
temperature = 0
top_k = 40
top_p = 0.9


start = time.perf_counter()

result = ollama.generate(
    model=model,
    prompt=prompt,
    stream=False,
    options={
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_predict": 200,
        "num_gpu":0
    }
)

elapsed = time.perf_counter() - start


print("Model:", model)
print("Temperature:", temperature)
print("Top-K:", top_k)
print("Top-P:", top_p)
print("\nResponse:\n", result["response"])
print(f"\nResponse time: {elapsed:.2f} seconds")
