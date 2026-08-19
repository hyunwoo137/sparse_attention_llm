import os
import sys
import torch
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from benchmark.ruler32k import Ruler32K

def main():
    print("=" * 80)
    print("Starting vAttention (Adaptive Sampling) Benchmark...")
    print("=" * 80)

    # 1. Config for vAttention (Adaptive Sampling)
    sparse_config = ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=1.0 / 32,
                epsilon=0.25,
                delta=0.25,
                init_offset=128,
                local_offset=128,
            ),
        ]
    )

    # 2. Select GPU device with free memory (cuda:3)
    device = "cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else "cuda:0"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    print(f"Loading Model: {model_name} on device: {device}...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    # 3. Setup MicroMetricLogger & Benchmark (RULER 32K niah_single_1)
    output_dir = repo_root / "results_vattention"
    output_dir.mkdir(exist_ok=True, parents=True)

    from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=str(output_dir),
        enabled_metrics=[
            "research_attention_density",
            "research_attention_output_error",
        ],
    )

    benchmark = Ruler32K(["niah_single_1"])

    print(f"Running Benchmark: RULER 32K (niah_single_1) -> Output: {output_dir}")
    benchmark.run_benchmark(
        adapter,
        str(output_dir),
        request_kwargs={"max_requests": 5, "max_context_length": 32000},
        generation_kwargs={"max_new_tokens": 50},
    )

    print("\n" + "=" * 80)
    print(f"vAttention Benchmark Completed Successfully!")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
