"""
Example: Inspect SALora search results
"""

import json
import sys

def analyze_salora_config(config_path):
    """Analyze and visualize SALora configuration"""

    print("="*80)
    print("SALora Configuration Analysis")
    print("="*80)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check format
    if 'salora_full_config' not in config:
        print("\nWarning: Old format detected. Please re-run search to get full configuration.")
        return

    full_config = config['salora_full_config']
    compression = config['compression_strategies']

    print(f"\n1. Search Results Overview")
    print(f"   - Total configurations: {len(full_config['layer_module_ranks'])}")
    print(f"   - Number of layers: {full_config['n_layers']}")
    print(f"   - Max rank searched: {full_config['search_params']['max_rank']}")

    print(f"\n2. Compression Strategies Summary")
    print(f"   - Global median: {compression['global_median']}")
    print(f"   - Global mean: {compression['global_mean']}")

    print(f"\n3. Per-Module-Type Analysis")
    print(f"   {'Module':<10} {'Median':<8} {'Mean':<8} {'Min':<6} {'Max':<6} {'Range'}")
    print(f"   {'-'*60}")

    for module_type, ranks in full_config['rank_summary_by_module_type'].items():
        import numpy as np
        median = compression['module_type_medians'][module_type]
        mean = compression['module_type_means'][module_type]
        min_r = min(ranks)
        max_r = max(ranks)

        print(f"   {module_type:<10} {median:<8} {mean:<8.1f} {min_r:<6} {max_r:<6} {ranks}")

    print(f"\n4. Layer Group Analysis")
    layer_groups = compression['layer_groups']
    for group_name in ['early_layers', 'middle_layers', 'late_layers']:
        group = layer_groups[group_name]
        indices = group['indices']
        median_rank = group['median_rank']
        print(f"   {group_name:15s}: layers {indices[0]:2d}-{indices[-1]:2d}, median rank = {median_rank}")

    print(f"\n5. Key Insights")

    # Find most/least important modules
    module_medians = compression['module_type_medians']
    sorted_modules = sorted(module_medians.items(), key=lambda x: x[1], reverse=True)

    print(f"   Most important modules:")
    for module, rank in sorted_modules[:3]:
        print(f"     - {module}: rank {rank}")

    print(f"   Least important modules:")
    for module, rank in sorted_modules[-3:]:
        print(f"     - {module}: rank {rank}")

    # Rank distribution
    all_ranks = [r for ranks in full_config['rank_summary_by_module_type'].values() for r in ranks]
    import numpy as np
    print(f"\n6. Overall Rank Distribution")
    print(f"   - Min: {min(all_ranks)}")
    print(f"   - 25th percentile: {int(np.percentile(all_ranks, 25))}")
    print(f"   - Median (50th): {int(np.median(all_ranks))}")
    print(f"   - 75th percentile: {int(np.percentile(all_ranks, 75))}")
    print(f"   - Max: {max(all_ranks)}")

    print(f"\n7. Recommendations for PEFT")
    print(f"   Based on the search results, we recommend:")
    print(f"   - Conservative: r={compression['layer_groups']['early_layers']['median_rank']} (early layers)")
    print(f"   - Balanced: r={compression['global_median']} (global median, recommended)")
    print(f"   - Aggressive: r={compression['layer_groups']['late_layers']['median_rank']} (late layers)")

    print(f"\n8. Module Selection Strategy")
    print(f"   If you want to further reduce parameters, consider:")

    # Suggest which modules to keep
    high_rank_modules = [m for m, r in module_medians.items() if r >= compression['global_median']]
    low_rank_modules = [m for m, r in module_medians.items() if r < compression['global_median']]

    print(f"   - Keep these modules (high rank): {', '.join(high_rank_modules)}")
    print(f"   - Consider pruning: {', '.join(low_rank_modules)}")

    print(f"\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_salora_results.py <path_to_peft_config.json>")
        print("\nExample:")
        print("  python analyze_salora_results.py ./output_code2nl/peft_config.json")
        sys.exit(1)

    analyze_salora_config(sys.argv[1])
