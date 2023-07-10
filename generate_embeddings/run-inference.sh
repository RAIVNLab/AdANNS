config=RR # RR or MRL

if config=RR
then
    for dim in 8 16 32 64 128 256 512 1024 2048
    do
        echo "Generating embeddings on RR-$dim"
        python pytorch_inference.py --retrieval --path=path/to/rr/model/weights.pt \
        --model_arch='resnet50' --retrieval_array_path=rr_output_dir/ --dataset=1K --rep_size=$dim
    done
else
    echo "Generating embeddings on MRL"
    python pytorch_inference.py --retrieval --path=path/to/mrl/model/weights.pt \
    --model_arch='resnet50' --retrieval_array_path=mrl_output_dir/ --dataset=1K --mrl
fi
