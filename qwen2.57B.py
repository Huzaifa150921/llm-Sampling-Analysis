
import ollama
import time


prompt = "Explain why the sky appears blue in simple terms."

model = "qwen2.5:7b"

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
