import itertools
import copy
import yaml
from utils.dict_utils import set_nested_key
from runners.train_core import train_from_config
import torch
import pandas as pd
from datetime import datetime
import os

def grid_search_runner(config_path, search_space, k=5):
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    keys, value_lists = zip(*search_space.items())
    combinations = list(itertools.product(*value_lists))
    all_results = []

    print("\n📊 Starting Grid Search over Hyperparameters")
    print("=============================================\n")

    for idx, combo in enumerate(combinations):
        print("────────────────────────────────────────────────────────────")
        print(f"🔍 Running Configuration {idx + 1} / {len(combinations)}")
        print("────────────────────────────────────────────────────────────")
        
        config = copy.deepcopy(base_config)
        config_summary = {}
        for key, val in zip(keys, combo):
            set_nested_key(config, key, val)
            config_summary[key] = val

        print("🔧 Hyperparameters:")
        for k, v in config_summary.items():
            print(f"   - {k}: {v}")

        print("\n🚀 Training...\n")
        metrics = train_from_config(config)
        all_results.append({
            "config": config_summary,
            "metrics": metrics
        })

    print("\n✅ Grid Search Complete")
    print("════════════════════════════════════════════════════════════")
    print("📈 Summary Table:")

    try:
        from tabulate import tabulate
        from metrics import load_metric

        metric_keys = list(all_results[0]["metrics"].keys())
        headers = list(search_space.keys()) + metric_keys
        raw_rows = []

        for result in all_results:
            param_values = [result["config"][k] for k in search_space.keys()]
            metric_values = [result["metrics"][m] for m in metric_keys]
            raw_rows.append(param_values + metric_values)

        # Determine whether each metric is to be maximized or minimized
        metric_directions = {}
        for name in metric_keys:
            _, _, _, direction = load_metric(name)(torch.tensor([0.0]), torch.tensor([0.0]))
            metric_directions[name] = direction

        # Find best indices per metric
        best_indices = {}
        for i, m in enumerate(metric_keys):
            col_idx = len(search_space) + i
            col = [row[col_idx] for row in raw_rows]
            direction = metric_directions[m]
            best_val = max(col) if direction == "max" else min(col)
            best_indices[m] = [j for j, val in enumerate(col) if val == best_val]

        # Print summary with terminal highlighting
        colored_rows = []
        for row_idx, row in enumerate(raw_rows):
            colored_row = []
            for col_idx, cell in enumerate(row):
                if col_idx >= len(search_space):  # metric column
                    metric_name = metric_keys[col_idx - len(search_space)]
                    if row_idx in best_indices[metric_name]:
                        cell = f"\033[1;32m *{cell:.4f}* \033[0m"
                    else:
                        cell = f"{cell:.4f}"
                else:
                    cell = str(cell)
                colored_row.append(cell)
            colored_rows.append(colored_row)

        print(tabulate(colored_rows, headers=headers, tablefmt="grid"))

    except ImportError:
        print("🔧 Install 'tabulate' for a pretty summary table: pip install tabulate")

    # 🔄 Save results to Excel with *best* values marked
    flat_results = []
    for row_idx, result in enumerate(all_results):
        row = {}
        row.update(result["config"])
        for m in metric_keys:
            val = result["metrics"][m]
            if row_idx in best_indices[m]:
                row[m] = f"*{val:.6f}*"
            else:
                row[m] = f"{val:.6f}"
        flat_results.append(row)

    df = pd.DataFrame(flat_results)

    # Construct descriptive filename
    model_type = base_config.get("model", {}).get("type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_desc = "_".join(
        f"{k.split('.')[-1]}={','.join(str(v) for v in search_space[k])}" 
        for k in search_space
    )
    safe_param_desc = param_desc.replace(".", "").replace(" ", "").replace("=", "-").replace(",", ",")

    filename = f"{model_type}__{safe_param_desc}__{timestamp}.xlsx"
    output_dir = "grid_search_results"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, filename)
    df.to_excel(excel_path, index=False)

    print(f"\n📁 Results saved to: {excel_path}")

    return all_results
