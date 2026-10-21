
import csv
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
    "capital_australia": {
        "prompt": "What is the capital of Australia? Answer with the city name only.",
        "type": "factual_known",
        "expected": ["canberra"],
    },
    "water_formula": {
        "prompt": "What is the chemical formula of water? Answer with the formula only.",
        "type": "factual_known",
        "expected": ["h2o"],
    },
    "pakistan_capital_language": {
        "prompt": "What is the capital of Pakistan and its national language? Answer in one sentence.",
        "type": "factual_known",
        "expected": ["islamabad", "urdu"],
    },
    "future_event": {
        "prompt": "Who won the 2030 FIFA World Cup? If the result is not yet known, say that it has not happened yet. Do not guess.",
        "type": "factual_unknown",
    },
    "creative": {
        "prompt": "Write a short, original story about a robot that discovers a garden on Mars.",
        "type": "creative",
    },
}

EXPERIMENTS = {
    "Temperature": [
        {"label": "Low (0.0)", "temperature": 0.0, "top_k": BASE_TOP_K, "top_p": BASE_TOP_P},
        {"label": "High (0.8)", "temperature": 0.8, "top_k": BASE_TOP_K, "top_p": BASE_TOP_P},
    ],
    "Top-K": [
        {"label": "Low (10)", "temperature": BASE_TEMPERATURE, "top_k": 10, "top_p": BASE_TOP_P},
        {"label": "High (80)", "temperature": BASE_TEMPERATURE, "top_k": 80, "top_p": BASE_TOP_P},
    ],
    "Top-P": [
        {"label": "Low (0.5)", "temperature": BASE_TEMPERATURE, "top_k": BASE_TOP_K, "top_p": 0.5},
        {"label": "High (0.95)", "temperature": BASE_TEMPERATURE, "top_k": BASE_TOP_K, "top_p": 0.95},
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


def remove_markdown_fence(response):
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_response(response):
    if not response or not response.strip():
        return None, False
    try:
        return json.loads(remove_markdown_fence(response)), True
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, False


def validate_json_schema_and_values(data):
    return (
        isinstance(data, dict)
        and set(data.keys()) == {"name", "age", "city"}
        and data.get("name") == "Ali"
        and str(data.get("age")) == "20"
        and data.get("city") == "Lahore"
    )


def calculate_determinism(responses):
    normalized = [" ".join(r.lower().split()) for r in responses if r]
    if not normalized:
        return 0.0
    return Counter(normalized).most_common(1)[0][1] / len(normalized) * 100.0


def evaluate_math(response):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    return bool(numbers and float(numbers[-1]) == 23.0)


def evaluate_known_factual_response(response, expected_terms):
    normalized = re.sub(r"\s+", " ", response.lower()).strip()
    found = [term for term in expected_terms if term.lower() in normalized]
    correct = len(found) == len(expected_terms)
    return {
        "expected_terms": " | ".join(expected_terms),
        "found_terms": " | ".join(found),
        "factual_answer_correct": correct,
        "factual_error": not correct,
    }


def evaluate_unknown_factual_response(response):
    normalized = re.sub(r"\s+", " ", response.lower()).strip()
    uncertainty_patterns = [
        r"\bhas not happened\b",
        r"\bhasn't happened\b",
        r"\bnot happened yet\b",
        r"\bhas not taken place\b",
        r"\bnot yet known\b",
        r"\bunknown\b",
        r"\bcannot know\b",
        r"\bcan't know\b",
        r"\bcannot predict\b",
        r"\bcan't predict\b",
        r"\bno winner\b",
        r"\bhas not been held\b",
        r"\bhasn't been held\b",
        r"\bnot been held\b",
        r"\bnot yet occurred\b",
        r"\bhas not occurred\b",
        r"\bnot scheduled\b",
        r"\bi don't know\b",
        r"\bi do not know\b",
        r"\bwill take place\b",
        r"\bis scheduled for\b",
    ]
    expresses_uncertainty = any(
        re.search(pattern, normalized)
        for pattern in uncertainty_patterns
    )
    return {
        "expresses_uncertainty": expresses_uncertainty,
        "potential_fabrication": not expresses_uncertainty,
    }


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
        "json_syntax_validity": [],
        "json_schema_correctness": [],
        "math_accuracy": [],
        "known_factual_accuracy": [],
        "unknown_question_potential_fabrication": [],
        "errors": [],
        "response_records": [],
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
                    f"ERROR | Model: {model} | Prompt: {prompt_name} | "
                    f"Repeat: {repeat + 1} | {result['error']}"
                )

            results["times"].append(result["time"])

            record = {
                "model": model,
                "parameter": "",
                "configuration": settings["label"],
                "prompt_name": prompt_name,
                "repeat": repeat + 1,
                "prompt": prompt_info["prompt"],
                "response": response,
                "error": result["error"] or "",
                "response_time_seconds": result["time"],
            }

            prompt_type = prompt_info["type"]

            if prompt_type == "json":
                parsed_data, syntax_valid = parse_json_response(response)
                schema_correct = (
                    validate_json_schema_and_values(parsed_data)
                    if syntax_valid else False
                )
                results["json_syntax_validity"].append(syntax_valid)
                results["json_schema_correctness"].append(schema_correct)
                record["json_syntax_valid"] = syntax_valid
                record["json_schema_and_values_correct"] = schema_correct
                print(f"JSON syntax valid: {syntax_valid}")
                print(f"JSON schema/values correct: {schema_correct}")

            elif prompt_type == "math":
                math_correct = evaluate_math(response)
                results["math_accuracy"].append(math_correct)
                record["math_correct"] = math_correct

            elif prompt_type == "factual_known":
                factual_result = evaluate_known_factual_response(
                    response,
                    prompt_info["expected"],
                )
                results["known_factual_accuracy"].append(
                    factual_result["factual_answer_correct"]
                )
                record.update(factual_result)
                print(f"Known factual response: {response!r}")
                print(
                    "Known factual answer correct: "
                    f"{factual_result['factual_answer_correct']}"
                )

            elif prompt_type == "factual_unknown":
                unknown_result = evaluate_unknown_factual_response(response)
                results["unknown_question_potential_fabrication"].append(
                    unknown_result["potential_fabrication"]
                )
                record.update(unknown_result)
                print(f"Unknown/future factual response: {response!r}")
                print(
                    "Potential fabrication flag: "
                    f"{unknown_result['potential_fabrication']}"
                )

            prompt_responses.append(response)
            results["response_records"].append(record)

            print(
                f"Model: {model} | Prompt: {prompt_name} | "
                f"Repeat: {repeat + 1}"
            )
            print(f"Response: {response or '[No response]'}")
            print("-" * 70)

        results["responses"][prompt_name] = prompt_responses

    determinism_values = [
        calculate_determinism(values)
        for values in results["responses"].values()
        if values
    ]

    def percentage(values):
        return float(np.mean(values)) * 100.0 if values else 0.0

    return {
        "determinism": (
            float(np.mean(determinism_values))
            if determinism_values else 0.0
        ),
        "json_syntax_validity": percentage(
            results["json_syntax_validity"]
        ),
        "json_schema_correctness": percentage(
            results["json_schema_correctness"]
        ),
        "math_accuracy": percentage(results["math_accuracy"]),
        "known_factual_accuracy": percentage(
            results["known_factual_accuracy"]
        ),
        "unknown_question_potential_fabrication_rate": percentage(
            results["unknown_question_potential_fabrication"]
        ),
        "average_response_time": (
            float(np.mean(results["times"]))
            if results["times"] else 0.0
        ),
        "responses": results["responses"],
        "errors": results["errors"],
        "response_records": results["response_records"],
    }


def run_all_experiments():
    all_results = {}
    all_records = []

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

                metrics = run_single_experiment(model, configuration)

                for record in metrics["response_records"]:
                    record["parameter"] = parameter
                    all_records.append(record)

                all_results[model][parameter][label] = metrics

                print("\nMetrics")
                print(f"Determinism: {metrics['determinism']:.2f}%")
                print(f"JSON Syntax Validity: {metrics['json_syntax_validity']:.2f}%")
                print(f"JSON Schema/Value Correctness: {metrics['json_schema_correctness']:.2f}%")
                print(f"Math Accuracy: {metrics['math_accuracy']:.2f}%")
                print(f"Known Factual Accuracy: {metrics['known_factual_accuracy']:.2f}%")
                print(
                    "Potential Fabrication Rate on Unknown/Future Question: "
                    f"{metrics['unknown_question_potential_fabrication_rate']:.2f}%"
                )
                print(
                    f"Average Response Time: "
                    f"{metrics['average_response_time']:.2f} seconds"
                )
                print(f"Request errors: {len(metrics['errors'])}")

    csv_path = OUTPUT_DIR / "all_model_responses.csv"
    fieldnames = sorted({
        key for record in all_records for key in record.keys()
    })

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    json_path = OUTPUT_DIR / "experiment_metrics_and_responses.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    print(f"\nResponse CSV saved: {csv_path.resolve()}")
    print(f"Full results JSON saved: {json_path.resolve()}")

    return all_results


def plot_metric(all_results, metric_key, metric_title, filename):
    parameters = list(EXPERIMENTS.keys())
    fig, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(18, 5),
        squeeze=False,
    )
    axes = axes[0]
    bar_width = 0.35

    for axis, parameter in zip(axes, parameters):
        configurations = EXPERIMENTS[parameter]
        labels = [configuration["label"] for configuration in configurations]
        x_positions = np.arange(len(labels))

        for model_index, model in enumerate(MODELS):
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
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("Percentage (%)")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)

    fig.suptitle(metric_title)
    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
        "json_syntax_validity",
        "JSON Syntax Validity Across Sampling Parameters",
        "JSON_Syntax_Validity_model_comparison.png",
    )
    plot_metric(
        all_results,
        "json_schema_correctness",
        "JSON Schema and Value Correctness Across Sampling Parameters",
        "JSON_Schema_Value_Correctness_model_comparison.png",
    )
    plot_metric(
        all_results,
        "known_factual_accuracy",
        "Accuracy on Known Factual Questions",
        "Known_Factual_Accuracy_model_comparison.png",
    )
    plot_metric(
        all_results,
        "unknown_question_potential_fabrication_rate",
        "Potential Fabrication Flags on Unknown/Future Question",
        "Potential_Fabrication_Rate_model_comparison.png",
    )
    plot_metric(
        all_results,
        "math_accuracy",
        "Math Accuracy Across Sampling Parameters",
        "Math_Accuracy_model_comparison.png",
    )


def main():
    print("LLM Sampling Analysis")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Repeats per prompt: {REPEATS}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    total_calls = (
        len(MODELS)
        * sum(len(configurations) for configurations in EXPERIMENTS.values())
        * len(PROMPTS)
        * REPEATS
    )
    print(f"Expected Ollama generation calls: {total_calls}")

    all_results = run_all_experiments()
    create_graphs(all_results)

    print("\nExperiment completed.")
    print(
        "The potential-fabrication graph is a heuristic, not a definitive "
        "hallucination rate. Review saved responses before reporting results."
    )


if __name__ == "__main__":
    main()
