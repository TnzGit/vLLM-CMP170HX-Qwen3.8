"""CPU regression for the opt-in CMP mixed-FP8 page alignment."""

import os

import torch

from vllm.v1.core.kv_cache_utils import (
    _align_heterogeneous_attention_page_sizes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


def _specs() -> dict[str, KVCacheSpec]:
    return {
        "target": FullAttentionSpec(
            block_size=880,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.float8_e4m3fn,
            indexes_kv_by_block_stride=True,
        ),
        "draft": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            indexes_kv_by_block_stride=True,
        ),
        "mamba": MambaSpec(
            block_size=880,
            shapes=((1, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
    }


def main() -> int:
    specs = _specs()
    os.environ["VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES"] = "0"
    unchanged = _align_heterogeneous_attention_page_sizes(specs)
    assert unchanged == specs

    os.environ["VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES"] = "1"
    aligned = _align_heterogeneous_attention_page_sizes(specs)
    assert aligned["target"].block_size == 896
    assert aligned["draft"].block_size == 448
    assert aligned["mamba"].block_size == 896
    assert aligned["target"].page_size_padded is None
    assert aligned["draft"].page_size_padded is None
    assert aligned["target"].page_size_bytes == 1_835_008
    assert aligned["draft"].page_size_bytes == 1_835_008
    assert aligned["mamba"].page_size_bytes == specs["mamba"].page_size_bytes
    print(
        "mixed-FP8 page alignment: target=896 draft=448 mamba=896 "
        "attention_page=1835008 OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
