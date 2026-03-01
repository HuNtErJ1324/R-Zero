model_path=$1
run_id=$2
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=2 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
CUDA_VISIBLE_DEVICES=3 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &

# Wait for both servers to be ready before returning.
# The questioner training's reward function will fail with ConnectionRefusedError
# if it fires before the vLLM servers finish loading the model.
wait_for_port() {
    local port=$1
    echo "Waiting for vLLM scorer server on port $port..."
    until (echo > /dev/tcp/127.0.0.1/$port) 2>/dev/null; do
        sleep 5
    done
    echo "vLLM scorer server on port $port is ready."
}

wait_for_port 5000
wait_for_port 5001
