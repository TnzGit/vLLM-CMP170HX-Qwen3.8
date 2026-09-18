#!/usr/bin/env python3
"""Apply the experimental INT8-G64 bridge to an isolated vLLM install.

The script is deliberately fail-closed: it only edits the vLLM package under
the supplied root, creates one backup per file, and refuses to reapply against
unknown source.  It never touches a system service or model directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        return
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:100]!r}")
    backup = path.with_name(path.name + ".orig-int8g64")
    if not backup.exists():
        backup.write_text(text)
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", type=Path)
    args = ap.parse_args()
    root = args.vllm_root
    tu = root / "utils/torch_utils.py"
    cache = root / "config/cache.py"
    kv = root / "v1/kv_cache_interface.py"
    tr = root / "v1/attention/backends/triton_attn.py"

    # A previous interrupted run could have inserted a duplicate decorator;
    # normalize it before the idempotent anchors below.
    kv_text = kv.read_text()
    if "    @property\n    @property\n    def is_int8_g64" in kv_text:
        kv.write_text(kv_text.replace(
            "    @property\n    @property\n    def is_int8_g64",
            "    @property\n    def is_int8_g64",
            1,
        ))

    replace_once(
        tu,
        '    "int8_per_token_head": torch.int8,\n',
        '    "int8_per_token_head": torch.int8,\n    "int8_g64": torch.int8,\n',
        '    "int8_g64": torch.int8,',
    )
    replace_once(
        tu,
        '        or kv_cache_dtype == "nvfp4"\n',
        '        or kv_cache_dtype == "nvfp4"\n'
        '        or kv_cache_dtype == "int8_g64"\n',
        '        or kv_cache_dtype == "int8_g64"',
    )
    replace_once(
        cache,
        '    "int8_per_token_head",\n    "fp8_per_token_head",\n',
        '    "int8_per_token_head",\n    "int8_g64",\n    "fp8_per_token_head",\n',
        '    "int8_g64",\n',
    )
    replace_once(
        kv,
        '    KVARN = 7  # KVarN Hadamard+Sinkhorn tile quant, packed K+V per (block, head) tile\n',
        '    KVARN = 7  # KVarN Hadamard+Sinkhorn tile quant, packed K+V per (block, head) tile\n'
        '    INT8_G64 = 8  # E38/E39 signed int8 with four FP16 G=64 scales\n',
        '    INT8_G64 = 8',
    )
    replace_once(
        kv,
        '            KVQuantMode.INT8_PER_TOKEN_HEAD,\n            KVQuantMode.FP8_PER_TOKEN_HEAD,',
        '            KVQuantMode.INT8_PER_TOKEN_HEAD,\n            KVQuantMode.INT8_G64,\n            KVQuantMode.FP8_PER_TOKEN_HEAD,',
        '            KVQuantMode.INT8_G64,',
    )
    replace_once(
        kv,
        '    if kv_cache_dtype == "int8_per_token_head":\n        return KVQuantMode.INT8_PER_TOKEN_HEAD\n',
        '    if kv_cache_dtype == "int8_per_token_head":\n        return KVQuantMode.INT8_PER_TOKEN_HEAD\n'
        '    if kv_cache_dtype == "int8_g64":\n        return KVQuantMode.INT8_G64\n',
        '    if kv_cache_dtype == "int8_g64":',
    )
    replace_once(
        kv,
        '        if self.kv_quant_mode.is_per_token_head:\n            unpadded += (\n',
        '        if self.kv_quant_mode.is_per_token_head:\n'
        '            if (self.kv_quant_mode == KVQuantMode.INT8_G64\n'
        '                    and self.block_size == 64\n'
        '                    and self.head_size == 256\n'
        '                    and self.num_kv_heads == 4):\n'
        '                groups = (self.head_size + 63) // 64\n'
        '                unpadded += (2 * self.block_size * self.num_kv_heads * groups * get_dtype_size(torch.float16))\n'
        '                return unpadded\n'
        '            unpadded += (\n',
        '            if self.kv_quant_mode == KVQuantMode.INT8_G64:',
    )
    replace_once(
        kv,
        '    @property\n    @property\n    def is_int8_g64(self) -> bool:\n',
        '    @property\n    def is_int8_g64(self) -> bool:\n',
        '    def is_int8_g64(self) -> bool:',
    )
    replace_once(
        kv,
        '    def is_nvfp4(self) -> bool:\n',
        '    @property\n'
        '    def is_int8_g64(self) -> bool:\n'
        '        return self == KVQuantMode.INT8_G64\n\n'
        '    @property\n'
        '    def is_nvfp4(self) -> bool:\n',
        '    def is_int8_g64(self) -> bool:',
    )

    replace_once(
        tr,
        '        "int8_per_token_head",\n        "fp8_per_token_head",',
        '        "int8_per_token_head",\n        "int8_g64",\n        "fp8_per_token_head",',
        '        "int8_g64",',
    )
    replace_once(
        tr,
        '        if block_size % 16 != 0:\n            raise ValueError("Block size must be a multiple of 16.")\n',
        '        if block_size % 16 != 0:\n            raise ValueError("Block size must be a multiple of 16.")\n'
        '        if cache_dtype_str == "int8_g64":\n'
        '            if block_size != 64 or head_size != 256 or num_kv_heads != 4:\n'
        '                # Hybrid/GDN groups stay on the stock packed INT8 view.\n'
        '                padded_hs = head_size + 4\n'
        '                return (num_blocks, num_kv_heads, block_size, 2 * padded_hs)\n'
        '            return (num_blocks, num_kv_heads, block_size, head_size)\n',
        '        if cache_dtype_str == "int8_g64":',
    )
    replace_once(
        tr,
        '    _k_scale_cache: torch.Tensor | None = None\n    _v_scale_cache: torch.Tensor | None = None\n',
        '    _k_scale_cache: torch.Tensor | None = None\n    _v_scale_cache: torch.Tensor | None = None\n'
        '    _g64_cache: tuple[torch.Tensor, ...] | None = None\n'
        '    _g64_workspace: tuple | None = None\n\n'
        '    def _ensure_g64_cache(self, kv_cache: torch.Tensor):\n'
        '        if self._g64_cache is None:\n'
        '            from vllm.v1.attention.ops.int8_g64 import g64_views\n'
        '            self._g64_cache = g64_views(kv_cache)\n'
        '            logger.info_once("INT8-G64 cache views enabled (HND, block_size=64).")\n'
        '        return self._g64_cache\n',
        '    _g64_cache: tuple[torch.Tensor, ...] | None = None',
    )
    replace_once(
        tr,
        '        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)\n'
        '        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head\n',
        '        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)\n'
        '        self._g64_enabled = (\n'
        '            self._kv_quant_mode == KVQuantMode.INT8_G64\n'
        '            and self.num_kv_heads == 4\n'
        '            and self.head_size == 256\n'
        '        )\n'
        '        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head\n',
        '        self._g64_enabled = (',
    )
    replace_once(
        tr,
        '        if self._is_per_token_head_quant:\n            key_cache, value_cache = self._pth_key_value_caches(kv_cache)\n            k_scale_cache = self._k_scale_cache\n            v_scale_cache = self._v_scale_cache\n            q_descale = k_descale = v_descale = None\n',
        '        if self._g64_enabled:\n'
        '            key_cache, value_cache, k_scale_cache, v_scale_cache = self._ensure_g64_cache(kv_cache)\n'
        '            q_descale = k_descale = v_descale = None\n'
        '        elif self._is_per_token_head_quant:\n'
        '            key_cache, value_cache = self._pth_key_value_caches(kv_cache)\n'
        '            k_scale_cache = self._k_scale_cache\n'
        '            v_scale_cache = self._v_scale_cache\n'
        '            q_descale = k_descale = v_descale = None\n',
        '        if self._kv_quant_mode == KVQuantMode.INT8_G64:',
    )
    replace_once(
        tr,
        '        if self._is_per_token_head_quant:\n            key_cache, value_cache = self._pth_key_value_caches(kv_cache)\n            k_scale_cache = self._k_scale_cache\n            v_scale_cache = self._v_scale_cache\n            triton_reshape_and_cache_flash_per_token_head_quant(\n',
        '        if self._g64_enabled:\n'
        '            from vllm.v1.attention.ops.int8_g64 import reshape_and_cache_int8_g64\n'
        '            key_cache, value_cache, k_scale_cache, v_scale_cache = self._ensure_g64_cache(kv_cache)\n'
        '            reshape_and_cache_int8_g64(key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slot_mapping)\n'
        '            return\n'
        '        if self._is_per_token_head_quant:\n'
        '            key_cache, value_cache = self._pth_key_value_caches(kv_cache)\n'
        '            k_scale_cache = self._k_scale_cache\n'
        '            v_scale_cache = self._v_scale_cache\n'
        '            triton_reshape_and_cache_flash_per_token_head_quant(\n',
        '        if self._kv_quant_mode == KVQuantMode.INT8_G64:',
    )
    replace_once(
        tr,
        '        # syv patch: the split-KV Triton verify attention',
        '''        # E38 INT8-G64 whole-model verify bridge.  This is intentionally
        # before the legacy split-KV route: the E38 extension owns the packed
        # cache layout and supports one batched fixed-eight-token launch.
        if (
            _spec_attn_enabled()
            and self._g64_enabled
            and max_seqlen_q == 8
            and attn_metadata.causal
            and self.sliding_window == (-1, -1)
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and self.sinks is None
            and self.chunk_lookback == -1
            and mm_prefix_range_tensor is None
            and attn_metadata.rswa_prefix_lens is None
            and output_scale is None
        ):
            from vllm.v1.attention.ops.int8_g64 import run_e38_attention
            batch = int(cu_seqlens_q.shape[0] - 1)
            if 1 <= batch <= 4 and num_actual_tokens == batch * 8:
                q = query[:num_actual_tokens]
                if not q.is_contiguous():
                    q = q.contiguous()
                positions = (
                    attn_metadata.seq_lens[:batch, None].to(torch.int32) - 8
                    + torch.arange(8, device=q.device, dtype=torch.int32)[None, :]
                ).reshape(-1)
                valid_columns = attn_metadata.seq_lens[:batch].to(torch.int32)
                tables = block_table[:batch].to(torch.int32)
                if not tables.is_contiguous():
                    tables = tables.contiguous()
                split_count = min(32, max(4, (int(max_seqlen_k) + 4095) // 4096))
                if (
                    self._g64_workspace is None
                    or self._g64_workspace[0] != batch
                    or self._g64_workspace[1] != split_count
                ):
                    self._g64_workspace = (
                        batch,
                        split_count,
                        torch.empty((batch, split_count, 8, self.num_heads, self.head_size), dtype=torch.bfloat16, device=q.device),
                        torch.empty((batch, split_count, 8, self.num_heads), dtype=torch.float32, device=q.device),
                        torch.empty((batch, split_count, 8, self.num_heads), dtype=torch.float32, device=q.device),
                    )
                _, _, partial_acc, partial_m, partial_l = self._g64_workspace
                run_e38_attention(
                    q, positions, key_cache, value_cache, k_scale_cache, v_scale_cache,
                    tables, valid_columns,
                    output[:num_actual_tokens].view(batch, 8, self.num_heads, self.head_size),
                    partial_acc, partial_m, partial_l, self.scale, split_count,
                    int(max_seqlen_k),
                )
                return output

        # syv patch: the split-KV Triton verify attention''',
        '        # E38 INT8-G64 whole-model verify bridge',
    )
    print("INT8-G64 vLLM compatibility patch applied")


if __name__ == "__main__":
    main()
