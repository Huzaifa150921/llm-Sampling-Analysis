import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ollama

MODELS = ["llama3.1:8b", "qwen2.5:7b"]
REPEATS = 3
NUM_PREDICT = 200
NUM_GPU = 0

BASE_TEMPERATURE = 0.2
BASE_TOP_K = 40
BASE_TOP_P = 0.9

OUTPUT_DIR = Path("sampling_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = {
    "json": {
        "prompt": 'Return only valid JSON with exactly these keys: "name", "age", "city". Use name="Ali", age=20, city="Lahore".',
        "type": "json",
    },
    "math": {
        "prompt": "Calculate 15 + 8. Return only the answer.",
        "type": "math",
    },
    "factual": {
        "prompt": "What is the capital of Pakistan, and what is its national language? Answer in one sentence.",
        "type": "factual",
    },
    "creative": {
        "prompt": "Write a short, original story about a robot that discovers a garden on Mars.",
        "type": "creative",
    },
}

EXPERIMENTS = {
    "Temperature": [
        {
            "label": "Low (0.0)",
            "temperature": 0.0,
            "top_k": BASE_TOP_K,
            "top_p": BASE_TOP_P,
        },
        {
            "label": "High (0.8)",
            "temperature": 0.8,
            "top_k": BASE_TOP_K,
            "top_p": BASE_TOP_P,
        },
    ],
    "Top-K": [
        {
            "label": "Low (10)",
            "temperature": BASE_TEMPERATURE,
            "top_k": 10,
            "top_p": BASE_TOP_P,
        },
        {
            "label": "High (80)",
            "temperature": BASE_TEMPERATURE,
            "top_k": 80,
            "top_p": BASE_TOP_P,
        },
    ],
    "Top-P": [
        {
            "label": "Low (0.5)",
            "temperature": BASE_TEMPERATURE,
            "top_k": BASE_TOP_K,
            "top_p": 0.5,
        },
        {
            "label": "High (0.95)",
            "temperature": BASE_TEMPERATURE,
            "top_k": BASE_TOP_K,
            "top_p": 0.95,
        },
    ],
}


def generate_response(model, prompt, options):
    start_time = time.perf_counter()

    try:
        result = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options,
        )

        return {
            "response": result.get("response", "").strip(),
            "time": time.perf_counter() - start_time,
            "error": None,
        }

    except Exception as error:
        return {
            "response": "",
            "time": time.perf_counter() - start_time,
            "error": str(error),
        }


def validate_json_structure(response):
    try:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "age", "city"}
            and data.get("name") == "Ali"
            and str(data.get("age")) == "20"
            and data.get("city") == "Lahore"
        )

    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def evaluate_response(prompt_type, response):
    if not response:
        return {
            "json_valid": None,
            "math_correct": None,
            "factual_score": None,
        }

    json_valid = None
    math_correct = None
    factual_score = None

    if prompt_type == "json":
        json_valid = validate_json_structure(response)

    elif prompt_type == "math":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        math_correct = bool(
            numbers and float(numbers[-1]) == 23
        )

    elif prompt_type == "factual":
        normalized = response.lower()
        score = 0

        if "islamabad" in normalized:
            score += 50

        if "urdu" in normalized:
            score += 50

        factual_score = score

    return {
        "json_valid": json_valid,
        "math_correct": math_correct,
        "factual_score": factual_score,
    }


def calculate_determinism(responses):
    normalized = [
        " ".join(response.lower().split())
        for response in responses
        if response
    ]

    if not normalized:
        return 0.0

    most_common_count = Counter(normalized).most_common(1)[0][1]

    return (most_common_count / len(normalized)) * 100


def run_single_experiment(model, settings):
    options = {
        "temperature": settings["temperature"],
        "top_k": settings["top_k"],
        "top_p": settings["top_p"],
        "num_predict": NUM_PREDICT,
        "num_gpu": NUM_GPU,
    }

    results = {
        "responses": {},
        "times": [],
        "json_validity": [],
        "math_accuracy": [],
        "factual_scores": [],
        "errors": [],
    }

    for prompt_name, prompt_info in PROMPTS.items():
        prompt_responses = []

        for repeat in range(REPEATS):
            result = generate_response(
                model,
                prompt_info["prompt"],
                options,
            )

            response = result["response"]

            if result["error"]:
                results["errors"].append({
                    "prompt": prompt_name,
                    "repeat": repeat + 1,
                    "error": result["error"],
                })

                print(
                    f"ERROR | Model: {model} | "
                    f"Prompt: {prompt_name} | "
                    f"Repeat: {repeat + 1} | "
                    f"{result['error']}"
                )

            results["times"].append(result["time"])

            evaluation = evaluate_response(
                prompt_info["type"],
                response,
            )

            if prompt_info["type"] == "json":
                results["json_validity"].append(
                    bool(evaluation["json_valid"])
                )

            if evaluation["math_correct"] is not None:
                results["math_accuracy"].append(
                    evaluation["math_correct"]
                )

            if evaluation["factual_score"] is not None:
                results["factual_scores"].append(
                    evaluation["factual_score"]
                )

            prompt_responses.append(response)

            print(
                f"Model: {model} | "
                f"Prompt: {prompt_name} | "
                f"Repeat: {repeat + 1}"
            )
            print(f"Response: {response or '[No response]'}")
            print("-" * 70)

        results["responses"][prompt_name] = prompt_responses

    determinism_values = [
        calculate_determinism(responses)
        for responses in results["responses"].values()
        if responses
    ]

    determinism = (
        float(np.mean(determinism_values))
        if determinism_values
        else 0.0
    )

    json_validity = (
        float(np.mean(results["json_validity"])) * 100
        if results["json_validity"]
        else 0.0
    )

    math_accuracy = (
        float(np.mean(results["math_accuracy"])) * 100
        if results["math_accuracy"]
        else 0.0
    )

    factual_accuracy = (
        float(np.mean(results["factual_scores"]))
        if results["factual_scores"]
        else 0.0
    )

    return {
        "determinism": determinism,
        "json_validity": json_validity,
        "math_accuracy": math_accuracy,
        "factual_accuracy": factual_accuracy,
        "hallucination_proxy": 100.0 - factual_accuracy,
        "average_response_time": (
            float(np.mean(results["times"]))
            if results["times"]
            else 0.0
        ),
        "responses": results["responses"],
        "errors": results["errors"],
    }


def run_all_experiments():
    all_results = {}

    for model in MODELS:
        all_results[model] = {}

        for parameter, configurations in EXPERIMENTS.items():
            all_results[model][parameter] = {}

            for configuration in configurations:
                label = configuration["label"]

                print("\n" + "=" * 80)
                print(f"Model: {model}")
                print(f"Parameter: {parameter}")
                print(f"Configuration: {label}")
                print("=" * 80)

                metrics = run_single_experiment(
                    model,
                    configuration,
                )

                all_results[model][parameter][label] = metrics

                print("\nMetrics")
                print(
                    f"Determinism: "
                    f"{metrics['determinism']:.2f}%"
                )
                print(
                    f"JSON Structure Validity: "
                    f"{metrics['json_validity']:.2f}%"
                )
                print(
                    f"Math Accuracy: "
                    f"{metrics['math_accuracy']:.2f}%"
                )
                print(
                    f"Factual Keyword Accuracy: "
                    f"{metrics['factual_accuracy']:.2f}%"
                )
                print(
                    f"Factual Keyword Error Proxy: "
                    f"{metrics['hallucination_proxy']:.2f}%"
                )
                print(
                    f"Average Response Time: "
                    f"{metrics['average_response_time']:.2f} seconds"
                )
                print(
                    f"Request errors: {len(metrics['errors'])}"
                )

    return all_results


def plot_metric(
    all_results,
    metric_key,
    metric_title,
    filename,
):
    parameters = list(EXPERIMENTS.keys())

    fig, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(17, 5),
    )

    bar_width = 0.35
    x_positions = np.arange(2)

    for axis, parameter in zip(axes, parameters):
        for model_index, model in enumerate(MODELS):
            labels = [
                configuration["label"]
                for configuration in EXPERIMENTS[parameter]
            ]

            values = [
                all_results[model][parameter][label][metric_key]
                for label in labels
            ]

            values = [
                float(value) if np.isfinite(value) else 0.0
                for value in values
            ]

            offset = (
                model_index - (len(MODELS) - 1) / 2
            ) * bar_width

            bars = axis.bar(
                x_positions + offset,
                values,
                bar_width,
                label=model,
            )

            axis.bar_label(
                bars,
                labels=[f"{value:.1f}" for value in values],
                padding=2,
                fontsize=7,
            )

        axis.set_title(parameter)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [
                configuration["label"]
                for configuration in EXPERIMENTS[parameter]
            ],
            fontsize=8,
        )
        axis.set_ylabel("Percentage (%)")
        axis.set_ylim(0, 105)
        axis.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)

    fig.suptitle(metric_title)
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Graph saved: {output_path.resolve()}")


def create_graphs(all_results):
    plot_metric(
        all_results,
        "determinism",
        "Determinism Comparison Across Sampling Parameters",
        "Determinism_model_comparison.png",
    )

    plot_metric(
        all_results,
        "json_validity",
        "JSON Structure Validity Across Sampling Parameters",
        "JSON_Structure_Validity_model_comparison.png",
    )

    plot_metric(
        all_results,
        "hallucination_proxy",
        "Factual Keyword Error Rate Proxy Across Sampling Parameters",
        "Hallucination_Rate_model_comparison.png",
    )


def main():
    print("LLM Sampling Analysis")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Repeats per prompt: {REPEATS}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    all_results = run_all_experiments()
    create_graphs(all_results)

    print("\nExperiment completed.")
    print(
        f"All three graphs are saved in: "
        f"{OUTPUT_DIR.resolve()}"
    )
    print(
        "Note: the hallucination graph is only a "
        "factual-keyword error proxy."
    )


if __name__ == "__main__":
    main()