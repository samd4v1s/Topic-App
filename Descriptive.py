from pathlib import Path

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer

from colour_palette import apply_theme

apply_theme()

app = typer.Typer()


@app.command()
def main(
    data_file: Path = typer.Argument(help="Long-format CSV with columns 'Team', 'Scenario', and phases."),
    output: Path = typer.Argument(help="Directory to save outputs (chart and HTML table) to."),
) -> None:
    """Run descriptive analysis and generate team comparison charts.

    DATA_FILE should be a long-format CSV with columns 'Team', 'Scenario',
    and 'Phase 1' through 'Phase 5'.
    """
    output.mkdir(parents=True, exist_ok=True)
    # 1. Load and Aggregate Data
    df = pd.read_csv(data_file)

    phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    teams = df['Team'].unique()

    # --- NEW STEP: Group by Team and Scenario and take the average ---
    # This creates a DataFrame where each Team/Scenario pair is a single unique row
    df_avg = df.groupby(['Team', 'Scenario'])[phases].mean().reset_index()

    # 2. Preparation for Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    x = np.arange(len(teams))

    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

    # 3. Manual Stacked Plotting
    for i, team in enumerate(teams):
        for j, scenario in enumerate(['Max', 'Min']):
            offset = -bar_width/2 if scenario == 'Max' else bar_width/2

            # --- CHANGED: Use the aggregated df_avg instead of df ---
            # We use .iloc[0] here safely because groupby.mean() ensures there's only 1 row per team/scenario
            mask = (df_avg['Team'] == team) & (df_avg['Scenario'] == scenario)

            if not mask.any():
                continue  # Skip if a specific team doesn't have a Max or Min scenario

            row = df_avg[mask].iloc[0]
            bottom = 0

            for p_idx, phase in enumerate(phases):
                val = row[phase]
                # Add label only once for the legend
                label = phase if i == 0 and j == 0 else ""

                ax.bar(x[i] + offset, val, bar_width, bottom=bottom,
                       color=colors[p_idx], label=label)
                bottom += val

            # 4. Add "Max/Min" Label above each bar
            ax.text(x[i] + offset, bottom + 0.5, scenario.upper(),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. Formatting the Axis
    ax.set_xticks(x)
    ax.set_xticklabels(teams, fontsize=12, rotation=-45, ha='left') # Slightly larger for readability
    ax.set_ylabel('Average Weeks to Complete', fontsize=10)
    ax.set_title('Average Timeline Comparison by Team (Max vs Min)', fontsize=14)

    # Handle Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Phases", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output / 'team_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    #also create a html table of the average values for each team and scenario
    html_table = df_avg.to_html(index=False)
    with open(output / 'average_workload_table.html', 'w') as f:
        f.write(html_table)

    logging.info(f"Outputs saved to {output}")


if __name__ == "__main__":
    app()
