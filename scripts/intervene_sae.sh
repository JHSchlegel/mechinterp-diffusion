#!/bin/bash
set -e
cd ../mechinterp_diffusion/experiments


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



# =============================================================================
# EXPERIMENT 1: TRAJECTORY PLOT
#
# Goal: Visualize how a feature's effect changes over time
# This section will process all prompts at once for each trajectory condition.
# =============================================================================
echo "--- RUNNING TRAJECTORY EXPERIMENTS ---"


python intervene_sae.py --intervention_mode trajectory --hook_type add --prompts "${PROMPTS[@]}" --features 2420 --intervention_values 80.0 --save_activation_heatmap
echo "--- TRAJECTORY EXPERIMENTS COMPLETE ---"


# =============================================================================
# EXPERIMENT 2: GRID PLOT (Intervening at different timesteps)
#
# Goal: Compare features and activation strengths across diffusion steps.
# =============================================================================

# --------------------------------------------------------------------------- #
#                    Down 2.0 Low-Resolution Intervention                     #
# --------------------------------------------------------------------------- #
python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 3094 4255 810 1362 2420 \
    --intervention_values 35.0 50.0 65.0 80.0 95.0 \
    --seed 42


python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 3094 4255 810 1362 2420 \
    --intervention_values 35.0 50.0 65.0 80.0 95.0 \
    --timesteps 0 1 2 3 4 5 6 7 8 9 \
    --seed 42

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 3094 4255 810 1362 2420 \
    --intervention_values 35.0 50.0 65.0 80.0 95.0 \
    --timesteps 10 11 12 13 14 15 16 17 18 19 \
    --seed 42

python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 3094 4255 810 1362 2420 \
    --intervention_values 35.0 50.0 65.0 80.0 95.0 \
    --timesteps 20 21 22 23 24 \
    --seed 42


# --------------------------------------------------------------------------- #
#                    Down 2.0 High-Resolution Intervention                    #
# --------------------------------------------------------------------------- #
python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 2165 3094 4255 810 1362 2420 \
    --intervention_values 35.0 50.0 65.0 80.0 95.0 \
    --seed 42 \
    --height 768 --width 768


# --------------------------------------------------------------------------- #
#                      Up 1.1 Low-Resolution Intervention                     #
# --------------------------------------------------------------------------- #
python intervene_sae.py \
    --intervention_mode grid --hook_type add \
    --target_module "unet.up_blocks.1.attentions.1" \
    --sae_path "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282" \
    --prompts "Portrait of a stoic Roman emperor, profile view." \
    --features 702 4040 4350 11 453 1615 \
    --intervention_values 20.0 35.0 50.0 65.0 80.0 \
    --seed 42

#

# =============================================================================
# EXPERIMENT 3: TOP-K TRACE (Knockout Cascade Analysis)
#
# Goal: Knock out a key feature to analyze downstream causal effects.
# =============================================================================
echo "--- RUNNING TOP-K TRACE (KNOCKOUT) EXPERIMENT ---"
python intervene_sae.py \
    --intervention_mode topk_trace --hook_type scale \
    --prompts "a futuristic image of a knight" \
    --timesteps 0  \
    --intervention_values -1.0 \
    --topk_trace_k 10
echo "--- KNOCKOUT EXPERIMENT COMPLETE ---"


# =============================================================================
# EXPERIMENT 4: RECONSTRUCTION EXPERIMENTS
# =============================================================================

echo "--- RUNNING RECONSTRUCTION EXPERIMENTS ---"

# SAE paths and hook names for the two attention blocks
SAE_PATHS=(
    "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282"
    "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282"
)

HOOK_NAMES=(
    "unet.up_blocks.1.attentions.1"
    "unet.down_blocks.2.attentions.0"
)

# Run reconstruction for each SAE/hook combination
for idx in 0 1; do
    SAE_PATH="${SAE_PATHS[$idx]}"
    HOOK_NAME="${HOOK_NAMES[$idx]}"

    echo ""
    echo ">>> Running reconstruction for: $HOOK_NAME"
    echo ">>> SAE Path: $SAE_PATH"
    echo "-----------------------------------------------------"

    # Run reconstruction for all simple prompts at once
    python intervene_sae.py \
        --hook_type reconstruct \
        --intervention_mode trajectory \
        --sae_path "$SAE_PATH" \
        --num_prompts 5 \
        --target_module "$HOOK_NAME" \
        --height 768 --width 768
done

'
