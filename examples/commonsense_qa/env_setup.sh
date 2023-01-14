# Support cuda and rocm environment
# for cuda we test on V100 and A100
# for rocm we test on MI100, the image tested with building with pytorch-1.11.0-rocm5.1.3

GPU_ENV="$1"
if [ $GPU_ENV = "cuda" ] || [ $GPU_ENV = "rocm" ]; then
    echo "Installing $GPU_ENV environment"
    pip install -r "requirements_torch_${GPU_ENV}.txt"
    pip install -r requirements.txt
    pip install numpy tqdm -U
else
    echo "Unsupported environment $GPU_ENV"
fi