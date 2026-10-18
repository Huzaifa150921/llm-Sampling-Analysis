
import ollama
import time
import json
import re
import pandas as pd

# 1. Models and prompts
models = ["llama3.1:8b", "qwen2.5:7b"]
repeats = 3

prompts = {
    "JSON": 'Return only JSON: {"name":"Ali","age":25,"city":"Lahore"}',
    "Arithmetic": "3 notebooks cost $5 each and 4 pens cost $2 each. What is the total? Show the calculation.",
    "Factual": "What is the capital of Pakistan and its national language?",
    "Creative": "Write a short slogan for a student productivity app."
}

# 2. Sampling configurations
configs = [
    {"name": "Baseline", "temperature": 0, "top_k": 40, "top_p": 0.9},
    {"name": "High_Temperature", "temperature": 0.8, "top_k": 40, "top_p": 0.9},
    {"name": "Low_Top_K", "temperature": 0, "top_k": 10, "top_p": 0.9},
    {"name": "Low_Top_P", "temperature": 0, "top_k": 40, "top_p": 0.5}
]

results = []

# 3. Run experiments
for model in models:
    for config in configs:
        for prompt_name, prompt in prompts.items():
            for repeat in range(repeats):

                start = time.perf_counter()

                try:
                    output = ollama.generate(
                        model=model,
                        prompt=prompt,
                        stream=False,
                        options={
                            "temperature": config["temperature"],
                            "top_k": config["top_k"],
                            "top_p": config["top_p"],
                            "num_predict": 200,
                            "num_gpu": 0,
                            "seed": 42
                        }
                    )

                    answer = output["response"].strip()
                    error = ""

                except Exception as e:
                    answer = ""
                    error = str(e)

                elapsed = time.perf_counter() - start

                # 4. Evaluate JSON validity
                json_valid = None
                correct = None

                if prompt_name == "JSON":
                    try:
                        data = json.loads(answer)
                        json_valid = isinstance(data, dict)
                        correct = (
                            json_valid
                            and data.get("name") == "Ali"
                            and data.get("age") == 25
                            and data.get("city") == "Lahore"
                        )
                    except (json.JSONDecodeError, TypeError):
                        json_valid = False
                        correct = False

                # 5. Check arithmetic answer
                elif prompt_name == "Arithmetic":
                    correct = all(
                        re.search(pattern, answer)
                        for pattern in [r"\b23\b", r"\b15\b", r"\b8\b"]
                    )

                # 6. Basic factual check
                elif prompt_name == "Factual":
                    correct = (
                        "islamabad" in answer.lower()
                        and "urdu" in answer.lower()
                    )

                results.append({
                    "Model": model,
                    "Configuration": config["name"],
                    "Prompt": prompt_name,
                    "Repeat": repeat + 1,
                    "Temperature": config["temperature"],
                    "Top_K": config["top_k"],
                    "Top_P": config["top_p"],
                    "Time": round(elapsed, 3),
                    "JSON_Valid": json_valid,
                    "Correct": correct,
                    "Response": answer,
                    "Error": error
                })

                print(
                    model, config["name"], prompt_name,
                    "Run:", repeat + 1,
                    "Time:", round(elapsed, 2), "seconds"
                )
                print("Response:", answer)

# 7. Save all individual responses
df = pd.DataFrame(results)
df.to_csv("sampling_raw_results.csv", index=False)

# 8. Calculate evaluation metrics
summary = []

for (model, config_name), group in df.groupby(
    ["Model", "Configuration"]
):
    # Determinism: percentage matching the most common response
    consistency_scores = []

    for _, prompt_group in group.groupby("Prompt"):
        responses = prompt_group["Response"]
        responses = responses[responses != ""]

        if len(responses) > 0:
            most_common_count = responses.value_counts().max()
            consistency_scores.append(
                most_common_count / len(responses) * 100
            )

    determinism = (
        round(sum(consistency_scores) / len(consistency_scores), 2)
        if consistency_scores else None
    )

    # JSON validity percentage
    json_rows = group[group["Prompt"] == "JSON"]
    json_rate = json_rows["JSON_Valid"].mean() * 100

    # Accuracy on prompts with automatic checks
    checked = group[group["Correct"].notna()]
    accuracy = checked["Correct"].mean() * 100

    # Average response time
    avg_time = group["Time"].mean()

    summary.append({
        "Model": model,
        "Configuration": config_name,
        "Determinism_Percent": round(determinism, 2),
        "JSON_Validity_Percent": round(json_rate, 2),
        "Checked_Answer_Accuracy_Percent": round(accuracy, 2),
        "Average_Time_Seconds": round(avg_time, 3)
    })

summary = pd.DataFrame(summary)

# 9. Compare each configuration against its baseline
baseline = summary[summary["Configuration"] == "Baseline"]

for metric in [
    "Determinism_Percent",
    "JSON_Validity_Percent",
    "Checked_Answer_Accuracy_Percent",
    "Average_Time_Seconds"
]:
    baseline_values = baseline.set_index("Model")[metric].to_dict()

    summary[metric + "_Change"] = summary.apply(
        lambda row: round(
            row[metric] - baseline_values[row["Model"]], 2
        ),
        axis=1
    )

summary.to_csv("sampling_summary.csv", index=False)

print("\n========== FINAL SUMMARY ==========")
print(summary.to_string(index=False))

print("\nSaved files:")
print("sampling_raw_results.csv")
print("sampling_summary.csv")
