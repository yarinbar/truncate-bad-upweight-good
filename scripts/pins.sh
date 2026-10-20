# Hugging Face artifacts used by the paper runs: "<repo_id> <revision> <repo_type> <destination relative to data dir>".
# Sourced by download_models.sh and download_data.sh.
QRPO=skandermoalla/qrpo-paper
PIN_TULU="allenai/Llama-3.1-Tulu-3-8B-SFT f2a0b46b0cfda21003c6141b1ff837b7e165524d model shared/models/llama"
PIN_ARMORM="RLHFlow/ArmoRM-Llama3-8B-v0.1 eb2676d20da2f2d41082289d23c59b9f7427f955 model shared/models/armorm"
PIN_SKYWORK_LLAMA="Skywork/Skywork-Reward-V2-Llama-3.1-8B cba2f842f3f1af2f1b2f0d35e794d789976390c5 model HF_CACHE"
PIN_SKYWORK_QWEN="Skywork/Skywork-Reward-V2-Qwen3-8B 6f19fdefb933293d4898bdb59a96f7223d998659 model HF_CACHE"
PIN_MISTRAL_SFT="yarinbar/tup-mistral-7b-sft-magpieair UNPUBLISHED model shared/models/mistral-sft"
PIN_MISTRAL_BASE="mistralai/Mistral-7B-Instruct-v0.2 63a8b081895390a26e140280378bc85ec8bce07a model shared/models/mistral"
PIN_ALPACA="tatsu-lab/alpaca_eval 2edc6fad8be6b14ea7230aabfd08188da6b8b814 dataset ALPACA"
PIN_EVAL_ARTIFACTS="yarinbar/tup-eval-artifacts UNPUBLISHED dataset shared/tup-eval-artifacts"
PIN_LLAMA_UF_RANDOM="$QRPO-llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm 54fa79b9012aefc938530d4a34a94e3fbf16fce5 dataset shared/datasets-with-ref-rewards/merged/llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2random-armorm"
PIN_LLAMA_UF_BEST="$QRPO-llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm 6a0d270192a778239188bd956ecad2643bd4aef7 dataset shared/datasets-with-ref-rewards/merged/llama-nosft-ultrafeedback-armorm-temp1-ref50-offpolicy2best-armorm"
PIN_LLAMA_MA_RANDOM="$QRPO-llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm 922ffdf760062ee698731926a7c1e99f74b73efc dataset shared/datasets-with-ref-rewards/merged/llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm"
PIN_LLAMA_MA_BEST="$QRPO-llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm feb17ce7c61f5988d42fecc0b7ede932c852c095 dataset shared/datasets-with-ref-rewards/merged/llama-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm"
PIN_MISTRAL_MA_RANDOM="$QRPO-mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm 21cc46ef2ab9006fd1eef8018d4e79653a3226bb dataset shared/datasets-with-ref-rewards/merged/mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2random-armorm"
PIN_MISTRAL_MA_BEST="$QRPO-mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm 3e1c902fab6e9254da676b974dcc4a0338060072 dataset shared/datasets-with-ref-rewards/merged/mistral-sft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm"
PIN_SFT_DATA="$QRPO-mistral-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm 9333194668c68782171ffa57b7c47cb18f17bec0 dataset shared/datasets-with-ref-rewards/merged/mistral-nosft-magpieair-armorm-temp1-ref50-offpolicy2best-armorm"
