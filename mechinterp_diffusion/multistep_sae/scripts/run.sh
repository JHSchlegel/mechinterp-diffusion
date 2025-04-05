screen -S "collect_latents" bash -c "
CUDA_VISIBLE_DEVICES=0 accelerate launch collect_latents.py \
    --dataset_name "flickr30k" \
    --dataset_size 30000\
    --batch_size 50\
    --model_name "CompVis/stable-diffusion-v1-4" \
    --hook_names "unet.up_blocks.1.attentions.1"
"
