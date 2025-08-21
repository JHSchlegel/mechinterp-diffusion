# #!/bin/bash
# cd ../mechinterp_diffusion/experiments

# # Feature intervention example
# # python intervene_sae.py \
# #     --features 2958 \
# #     --intervention_values 15.0 \
# #     --hook_type scale \
# #     --seed 42 \
# #     --intervention_mode trajectory \
# #     --save_activation_heatmap \
# #     --prompts "A pair of black boots with a silver buckle."

# # Grid intervention example
# # python intervene_sae.py \
# #     --intervention_mode grid \
# #     --hook_type scale \
# #     --features 1756 2818 3008 \
# #     --prompts "A raccoon wearing a tuxedo." \
# #     --intervention_values 10.0 25.0 50.0 200.0 \
# #     --timesteps 19 20 21 22 23 24

# python intervene_sae.py \
#     --intervention_mode topk_trace \
#     --timesteps 10 \
#     --topk_trace_k 10 \
#     --intervention_values 0.0 \
#     --seed 42 \
#     --prompts "A majestic lion in the savanna"

#!/bin/bash
# ===================================================================================
# Minimalist Bash Script for SAE Feature Intervention Analysis
#
# A simplified script for generating paper-ready figures with SD 2.1.
# Assumes a 25-step diffusion process.
#
# How to use:
# 1. Customize the FEATURE_ID and other parameters under "Configuration".
# 2. Uncomment the block for the experiment you want to run.
# 3. Execute the script from your shell: ./run_interventions.sh
#
# ===================================================================================

# --- Global Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
cd ../mechinterp_diffusion/experiments

# --- Key Parameters to Configure ---

# --- Prompts & Features ---
# A list of high-quality prompts for the trajectory analysis loop.
PROMPTS=(
    "A high-resolution photograph of an astronaut standing on the moon."
    "A realistic image of a close-up shot of a pirate."
    "A realistic photograph of a raccoon wearing a tuxedo."
    "A fantasy portrait of a knight in shining armor."
    "A vintage steam locomotive racing through countryside meadows."
    "A peaceful zen garden with cherry blossoms and stone bridges."
    "A cozy cottage with a fireplace in a snowy winter forest."
    "A fierce dragon soaring above medieval castle towers."
    "An ancient Egyptian pyramid under starlit desert skies."
    "A sci-fi portrait of a robot with glowing circuits."
    "A dramatic volcanic landscape with lava flows and ash clouds."
    "A sleek spaceship cruising through colorful nebula clouds."
    "A modern glass house overlooking a tropical beach paradise."
    "A graceful phoenix rising from glowing crystal formations."
    "A floating city with waterfalls cascading into cloudy atmosphere."
)

# # TODO: Replace with a feature ID you are investigating.
FEATURE_ID=2165
# # TODO: Replace with a list of features for the grid plot.
# GRID_FEATURES="1756 2887 3008"


# ===================================================================================
# EXPERIMENT 1: TRAJECTORY PLOT (Comparing Intervention Timings)
#
# Goal: Visualize how a feature's effect changes based on when it's applied.
# This section will process all prompts at once for each trajectory condition.
# ===================================================================================
echo "--- RUNNING TRAJECTORY EXPERIMENTS ---"

# --- Condition 1.1: Timesteps 0-4 ---
# echo "  -> Intervening at steps 0-4 for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --timesteps $(seq 0 4) --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# # --- Condition 1.2: Timesteps 5-9 ---
# echo "  -> Intervening at steps 5-9 for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --timesteps $(seq 5 9) --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# # --- Condition 1.3: Timesteps 10-14 ---
# echo "  -> Intervening at steps 10-14 for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --timesteps $(seq 10 14) --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# # --- Condition 1.4: Timesteps 15-19 ---
# echo "  -> Intervening at steps 15-19 for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --timesteps $(seq 15 19) --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# # --- Condition 1.5: Timesteps 20-24 ---
# echo "  -> Intervening at steps 20-24 for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --timesteps $(seq 20 24) --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# --- Condition 1.6: Full Intervention (Constant Effect) ---
# echo "  -> Intervening at ALL steps for all prompts..."
# python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features $FEATURE_ID --intervention_values 20.0 --save_activation_heatmap --seed $SEED --height $HEIGHT --width $WIDTH

# echo "--- TRAJECTORY EXPERIMENTS COMPLETE ---"


# =============================================================================
# EXPERIMENT 2: GRID PLOT (Intervening at all timesteps)
#
# Goal: Compare features and activation strengths across all diffusion steps.
# =============================================================================

# --------------------------------------------------------------------------- #
#                    Down 2.0 Low-Resolution Intervention                     #
# --------------------------------------------------------------------------- #
# python intervene_sae.py \
#     --intervention_mode grid --hook_type add \
#     --prompts "Portrait of a stoic Roman emperor, profile view." \
#     --features 151 2165 4255 810 1362 2420 \
#     --intervention_values 20.0 35.0 50.0 65.0 80.0 \
#     --seed 42

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42 \
    --height 768 --width 768

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 3094 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42 \
    --height 768 --width 768

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 4255 \
    --intervention_values 25.0 50.0 75.0 100.0 200.0 \
    --seed 42 \
    --height 768 --width 768

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 810 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42 \
    --height 768 --width 768

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 1362 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42 \
    --height 768 --width 768

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2420 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42 \
    --height 768 --width 768

# --------------------------------------------------------------------------- #
#                    Down 2.0 High-Resolution Intervention                    #
# --------------------------------------------------------------------------- #
# python intervene_sae.py \
#     --intervention_mode grid --hook_type add \
#     --prompts "Portrait of a stoic Roman emperor, profile view." \
#     --features 151 2165 4255 810 1362 2420 \
#     --intervention_values 20.0 35.0 50.0 65.0 80.0 \
#     --seed 42 \
#     --height 768 --width 768

# --------------------------------------------------------------------------- #
#                      Up 1.1 Low-Resolution Intervention                     #
# --------------------------------------------------------------------------- #
# python intervene_sae.py \
#     --intervention_mode grid --hook_type add \
#     --target_module "unet.up_blocks.1.attentions.1" \
#     --sae_path "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282" \
#     --prompts "Portrait of a stoic Roman emperor, profile view." \
#     --features 702 4040 4350 11 453 1615 \
#     --intervention_values 20.0 35.0 50.0 65.0 80.0 \
#     --seed 42

#

# =============================================================================
# EXPERIMENT 3: TOP-K TRACE (Knockout Cascade Analysis)
#
# Goal: Knock out a key feature to analyze downstream causal effects.
# =============================================================================
# echo "--- RUNNING TOP-K TRACE (KNOCKOUT) EXPERIMENT ---"
# python intervene_sae.py \
#     --intervention_mode topk_trace --hook_type scale \
#     --prompts "a futuristic image of a knight" \
#     --timesteps 0  \
#     --intervention_values -1.0 \
#     --topk_trace_k 10 \
#     --seed $SEED --height $HEIGHT --width $WIDTH
# echo "--- TOP-K TRACE EXPERIMENT COMPLETE ---"


# =============================================================================
# EXPERIMENT 4: RECONSTRUCTION EXPERIMENTS
#
# Goal: Test SAE reconstruction quality with simple LAION-COCO style prompts
# Testing both attention blocks and their respective SAEs
# =============================================================================

# screen -S reconstruction_experiment bash -c '
# echo "--- RUNNING RECONSTRUCTION EXPERIMENTS ---"

# # SAE paths and hook names for the two attention blocks
# SAE_PATHS=(
#     "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282"
#     "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282"
# )

# HOOK_NAMES=(
#     "unet.up_blocks.1.attentions.1"
#     "unet.down_blocks.2.attentions.0"
# )

# # Run reconstruction for each SAE/hook combination
# for idx in 0 1; do
#     SAE_PATH="${SAE_PATHS[$idx]}"
#     HOOK_NAME="${HOOK_NAMES[$idx]}"

#     echo ""
#     echo ">>> Running reconstruction for: $HOOK_NAME"
#     echo ">>> SAE Path: $SAE_PATH"
#     echo "-----------------------------------------------------"

#     # Run reconstruction for all simple prompts at once
#     python intervene_sae.py \
#         --hook_type reconstruct \
#         --intervention_mode trajectory \
#         --sae_path "$SAE_PATH" \
#         --num_prompts 5 \
#         --target_module "$HOOK_NAME" \
#         --height 768 --width 768
# done

# echo "--- RECONSTRUCTION EXPERIMENTS COMPLETE ---"
# '

        #--prompts "${SIMPLE_PROMPTS[@]}" \
