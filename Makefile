# Directories for input data and outputs
DATA_DIR := data_synthetic
OUTPUT_DIR := output

# Declare targets that don't produce a file of the same name
.PHONY: all test clean

# Default target: run both pipelines
all: simulation descriptive

# --- Monte Carlo simulation pipeline ---
# Depends on the scripts, shared modules, and the wide-format CSV.
# Only reruns if any of these files change.
simulation: $(OUTPUT_DIR)/simulation/systemic_error_multiplier.png

$(OUTPUT_DIR)/simulation/systemic_error_multiplier.png: MC_simulation.py simulation.py data.py colour_palette.py $(DATA_DIR)/Raw-data-wide.csv
	uv run python MC_simulation.py $(DATA_DIR)/Raw-data-wide.csv $(OUTPUT_DIR)/simulation

# --- Descriptive analysis pipeline ---
# Depends on the script, colour palette, and the long-format CSV.
descriptive: $(OUTPUT_DIR)/descriptive/team_comparison.png

$(OUTPUT_DIR)/descriptive/team_comparison.png: Descriptive.py colour_palette.py $(DATA_DIR)/Raw-data-long.csv
	uv run python Descriptive.py $(DATA_DIR)/Raw-data-long.csv $(OUTPUT_DIR)/descriptive

# --- Utility targets ---

# Run the test suite
test:
	uv run pytest tests/ -v

# Remove all generated outputs
clean:
	rm -rf $(OUTPUT_DIR)
