# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python scripts/model_merger.py merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

import argparse
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PretrainedConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm

from verl.utils import hf_processor, hf_tokenizer


@dataclass
class ModelMergerConfig:
    operation: str  # 'merge' or 'test'
    backend: str
    local_dir: str
    hf_model_config_path: str
    target_dir: Optional[str] = "tmp"
    hf_upload_path: Optional[str] = None
    private: bool = False
    test_hf_dir: Optional[str] = None
    tie_word_embedding: bool = False
    is_value_model: bool = False
    hf_model_path: Optional[str] = None
    hf_upload: bool = field(init=False)

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


class BaseModelMerger(ABC):
    def __init__(self, config: ModelMergerConfig):
        self.config = config
        self.hf_model_config_path = config.hf_model_config_path

        if config.hf_model_path:
            print("Warning: --hf_model_path is deprecated and will be removed in a future version. Currently verl will save huggingface model configuration files into checkpoint directories. Therefore, there is no need to provide --hf_model_path. ")
            self.hf_model_config_path = config.hf_model_path

        self.model_config = AutoConfig.from_pretrained(self.hf_model_config_path)

    def get_transformers_auto_model_class(self):
        if "ForTokenClassification" in self.model_config.architectures[0]:
            return AutoModelForTokenClassification
        elif "ForCausalLM" in self.model_config.architectures[0]:
            return AutoModelForCausalLM
        elif "ForConditionalGeneration" in self.model_config.architectures[0]:
            return AutoModelForVision2Seq

        raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            except OSError:
                print(f"Warning: Generation config file not found in {self.hf_model_config_path}, using a generation config created from the model config.")
        return model

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(self.model_config, torch_dtype=torch.bfloat16)
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path)
        tokenizer = hf_tokenizer(self.hf_model_config_path)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

    def upload_to_huggingface(self):
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")

    @abstractmethod
    def merge_and_save(self):
        raise NotImplementedError("Subclasses should implement this method")


class FSDPModelMerger(BaseModelMerger):
    def _get_world_size(self) -> int:
        """Extracts the FSDP world_size from checkpoint filenames (e.g., 'model_world_size_8_rank_0.pt')."""
        for filename in os.listdir(self.config.local_dir):
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
            if match:
                return int(match.group(1))
        raise FileNotFoundError(f"Could not determine world size. No file matching 'model_world_size_(\d+)_rank_0.pt' found in {self.config.local_dir}")

    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        return torch.load(Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt", map_location="cpu", weights_only=False)

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        pivot_key = sorted(list(state_dict.keys()))[0]
        weight = state_dict[pivot_key]

        if isinstance(weight, DTensor):
            # get sharding info
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            # for non-DTensor
            mesh = np.array([world_size], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        return mesh, mesh_dim_names

    def _calculate_shard_configuration(self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
        """Calculates the total number of shards and the shape of the device mesh."""
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merges a list of tensors based on their DTensor placement"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # Merge state dicts from all shards
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        del model_state_dict_lst

        # Merge tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                # merge shards
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def merge_and_save(self):
        world_size = self._get_world_size()
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._test_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()

        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()
        del hf_model

        hf_model_keys = set(hf_state_dict.keys())
        collected_keys = set(state_dict.keys())

        missing_keys = hf_model_keys - collected_keys
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

        extra_keys = collected_keys - hf_model_keys
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

        for key in hf_model_keys:
            hf_shape = hf_state_dict[key].shape
            collected_shape = state_dict[key].shape
            assert hf_shape == collected_shape, f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"

            hf_dtype = hf_state_dict[key].dtype
            collected_dtype = state_dict[key].dtype
            assert hf_dtype == collected_dtype, f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"

            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)

        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")


class MegatronModelMerger(BaseModelMerger):
    """
    Merge Megatron-style checkpoints into a single Hugging Face model.
    During merging we log every parameter key we encounter so you can later
    create a definitive Megatron→Qwen name map.  For now the hook just
    records keys and returns them unchanged.
    """

    # ------------------------------------------------------------------ #
    #                            constructor                             #
    # ------------------------------------------------------------------ #
    def __init__(self, config: ModelMergerConfig):
        from verl.utils.megatron_utils import get_hf_config_and_tokenizer_checkpoint_path

        # Locate the HF config inside the Megatron checkpoint tree
        config.hf_model_config_path = get_hf_config_and_tokenizer_checkpoint_path(
            config.local_dir
        )
        super().__init__(config)

        # Track every param name we see during merge
        self._encountered_param_names: Set[str] = set()

    # ------------------------------------------------------------------ #
    #                          helper hooks                              #
    # ------------------------------------------------------------------ #
    def rename_while_merging(self, megatron_name: str) -> str:
        """
        Placeholder hook – currently just logs the name and returns it
        unchanged.  Extend this to perform real renaming when your Qwen
        mapping is ready.
        """
        self._encountered_param_names.add(megatron_name)
        return megatron_name

    def _print_encountered_names(self) -> None:
        """Pretty-print the full set of Megatron keys we touched."""
        if not self._encountered_param_names:
            print("No parameter names were recorded.")
            return
        print("\n--- Megatron parameter names encountered during merge ---")
        for name in sorted(self._encountered_param_names):
            print(name)
        print("--- End of list ---\n")

    # ------------------------------------------------------------------ #
    #                     checkpoint structure helpers                   #
    # ------------------------------------------------------------------ #
    def _get_tp_pp_rank_from_sharded_dir(self, sharded_dir: str) -> tuple[int, int]:
        m = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
        assert m, f"Invalid sharded dir {sharded_dir}"
        return int(m.group(1)), int(m.group(2))

    def _check_megatron_checkpoint_path(self, model_path: str) -> tuple[list[str], int, int]:
        tp_size = 0
        pp_size = 0
        sharded_dirs = sorted(os.listdir(model_path))
        for d in sharded_dirs:
            assert "model.pt" in os.listdir(Path(model_path) / d), f"model.pt missing in {d}"
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(d)
            tp_size = max(tp_size, tp_rank + 1)
            pp_size = max(pp_size, pp_rank + 1)
        return sharded_dirs, tp_size, pp_size

    # ------------------------------------------------------------------ #
    #            tensor concatenation rules across TP partitions         #
    # ------------------------------------------------------------------ #
    def _merge_across_tp(
        self,
        key: str,
        tp_data: list[torch.Tensor],
        config: PretrainedConfig,
        tp_size: int,
        is_value_model: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        if "linear_fc1.weight" in key:
            # gate + up in a single weight: split then concat by gate/up
            gate, up = zip(*(t.chunk(2) for t in tp_data))
            return [torch.cat(gate, 0), torch.cat(up, 0)]

        if "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # qkv packed together – split per TP first, then per head type
            q, k, v = [], [], []
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            kv_size_per_tp = tp_data[0].shape[0] // (num_q_per_kv + 2)

            for t in tp_data:
                num_q_groups = config.num_key_value_heads // tp_size
                for chunk in t.chunk(num_q_groups):
                    split_sizes = [
                        kv_size_per_tp * num_q_per_kv // num_q_groups,
                        kv_size_per_tp // num_q_groups,
                        kv_size_per_tp // num_q_groups,
                    ]
                    q_i, k_i, v_i = chunk.split(split_sizes)
                    q.append(q_i)
                    k.append(k_i)
                    v.append(v_i)
            return [torch.cat(q, 0), torch.cat(k, 0), torch.cat(v, 0)]

        if ("layer_norm" in key or "layernorm" in key or "output_layer" in key) and is_value_model:
            return tp_data[0]

        dim = 1 if ("linear_fc2.weight" in key or "self_attention.linear_proj" in key) else 0
        return torch.cat(tp_data, dim=dim)

    # ------------------------------------------------------------------ #
    #                    loading individual TP/PP shards                 #
    # ------------------------------------------------------------------ #
    def _load_state_dicts(
        self,
        model_ckpt_path: str,
        sharded_dirs: list[str],
        tp_size: int,
        pp_size: int,
    ) -> list[list[dict]]:
        sd_grid = [[None] * tp_size for _ in range(pp_size)]

        def _load_one(d: str):
            sd = torch.load(Path(model_ckpt_path) / d / "model.pt", map_location="cpu", weights_only=False)
            tp_rank, pp_rank = self._get_tp_pp_rank_from_sharded_dir(d)
            sd_grid[pp_rank][tp_rank] = sd

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as ex:
            list(
                tqdm(
                    ex.map(_load_one, sharded_dirs),
                    total=len(sharded_dirs),
                    desc=f"Loading {len(sharded_dirs)} Megatron shards",
                )
            )
        return sd_grid

    # ------------------------------------------------------------------ #
    #                    merging shards across TP / PP                   #
    # ------------------------------------------------------------------ #
    def _merge_state_dicts(
        self,
        sd_grid: list[list[dict]],
        tp_size: int,
        pp_size: int,
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        vpp_size = len(sd_grid[0][0])          # virtual pipeline stages
        layers_cum = 0

        for vpp in range(vpp_size):
            for pp in range(pp_size):
                handled = 0
                keys = sd_grid[pp][0][vpp].keys()
                for key in keys:
                    if "extra_state" in key:
                        continue
                    if self.config.tie_word_embedding and "output_layer" in key:
                        print("skip lm_head/reward_head due to tie-word-embedding")
                        continue

                    new_key = key
                    if "decoder.layers." in key:
                        local_layer = int(key.split(".")[2])
                        handled = max(local_layer, handled)
                        global_layer = local_layer + layers_cum
                        parts = key.split(".")
                        parts[2] = str(global_layer)
                        new_key = ".".join(parts)

                    # ---- hook: record & (eventually) rename ------------
                    new_key = self.rename_while_merging(new_key)

                    tp_tensors = [sd_grid[pp][tp][vpp][key] for tp in range(tp_size)]
                    merged_tensor = self._merge_across_tp(
                        new_key, tp_tensors, self.model_config, tp_size, self.config.is_value_model
                    )

                    if isinstance(merged_tensor, list):
                        if len(merged_tensor) == 3:        # qkv split
                            for n, t in zip(("q", "k", "v"), merged_tensor):
                                merged[new_key.replace("linear_qkv", f"linear_{n}")] = t
                        else:                              # gate + up
                            merged[new_key.replace("linear_fc1", "gate_proj")] = merged_tensor[0]
                            merged[new_key.replace("linear_fc1", "up_proj")] = merged_tensor[1]
                    else:
                        merged[new_key] = merged_tensor

                layers_cum += handled + 1

        return merged

    # ------------------------------------------------------------------ #
    #                      public entry-point                            #
    # ------------------------------------------------------------------ #
    def merge_and_save(self):
        from verl.utils.megatron_utils import get_model_checkpoint_path

        ckpt_path = get_model_checkpoint_path(self.config.local_dir)
        sharded_dirs, tp_size, pp_size = self._check_megatron_checkpoint_path(ckpt_path)
        print(
            f"Found {len(sharded_dirs)} shards • tp={tp_size} • pp={pp_size}"
        )

        sd_grid = self._load_state_dicts(ckpt_path, sharded_dirs, tp_size, pp_size)
        merged_state_dict = self._merge_state_dicts(sd_grid, tp_size, pp_size)
        del sd_grid

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("--test_hf_dir is required for test operation")
            self._test_state_dict(merged_state_dict)

        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            # Print the catalog of names we saw
            self._print_encountered_names()
            if self.config.hf_upload:
                self.upload_to_huggingface()

        else:
            raise ValueError(f"Unknown operation {self.config.operation}")

    # ------------------------------------------------------------------ #
    #           test / name-replacement helpers (unchanged)              #
    # ------------------------------------------------------------------ #
    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        ref_state_dict = load_file(Path(self.config.test_hf_dir) / "model.safetensors")

        params_mapping = [
            ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
            ("self_attention.linear_qkv.layer_norm_bias",  "input_layernorm.bias"),
            ("embedding.word_embeddings",                  "model.embed_tokens"),
            ("self_attention.linear_qkv",                  "self_attn.qkv_proj"),
            ("self_attention.linear_proj",                 "self_attn.o_proj"),
            ("pre_mlp_layernorm",                          "post_attention_layernorm"),
            ("mlp.linear_fc1.layer_norm_weight",           "post_attention_layernorm.weight"),
            ("mlp.linear_fc1.layer_norm_bias",             "post_attention_layernorm.bias"),
            ("mlp.linear_fc1",                             "mlp.gate_up_proj"),
            ("mlp.linear_fc2",                             "mlp.down_proj"),
            ("decoder.final_layernorm",                    "model.norm"),
            ("output_layer",                               "lm_head"),
            ("self_attention.linear_q",                    "self_attn.q_proj"),
            ("self_attention.linear_k",                    "self_attn.k_proj"),
            ("self_attention.linear_v",                    "self_attn.v_proj"),
        ]

        for orig_name, tensor in state_dict.items():
            name = self._replace_name(orig_name, params_mapping)
            if not name or (name.endswith(".bias") and name not in ref_state_dict):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embedding and "lm_head.weight" in name:
                continue
            if name not in ref_state_dict:
                raise RuntimeError(f"Key {name} missing in reference state dict")
            ref_tensor = ref_state_dict[name]
            assert tensor.dtype == ref_tensor.dtype
            torch.testing.assert_close(tensor, ref_tensor, atol=1e-2, rtol=5e-2)

    def _replace_name(self, megatron_name: str, mapping: list[tuple[str, str]]) -> Optional[str]:
        for m_src, v_dst in mapping:
            if m_src not in megatron_name:
                continue
            if "layers" in megatron_name:
                megatron_name = megatron_name.replace("decoder", "model")
                parts = megatron_name.split(".")
                if "layer_norm_weight" in parts or "layer_norm_bias" in parts:
                    return ".".join(parts[:3] + [v_dst])
                weight_or_bias = parts[-1]
                return ".".join(parts[:3] + [v_dst, weight_or_bias])
            return megatron_name.replace(m_src, v_dst)
        return None


def main():
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument("--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model")
    base_op_parser.add_argument("--local_dir", type=str, required=True, help="Path to the saved model checkpoints")
    base_op_parser.add_argument("--hf_model_path", type=str, default=None, help="(Deprecated) Path to the original Hugging Face model for config.")
    base_op_parser.add_argument("--tie-word-embedding", action="store_true", help="Whether to tie word embedding weights (currently only Megatron supported)")
    base_op_parser.add_argument("--is-value-model", action="store_true", help="Whether the model is a value model (currently only Megatron supported)")

    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")
    merge_parser.add_argument("--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model")
    merge_parser.add_argument("--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model")
    merge_parser.add_argument("--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository")

    test_parser = subparsers.add_parser("test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model")
    test_parser.add_argument("--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing")

    args = parser.parse_args()

    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_path": args.hf_model_path,
        "hf_model_config_path": args.local_dir,
    }

    if args.operation == "merge":
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            private=args.private,
            test_hf_dir=None,
        )
        os.makedirs(config.target_dir, exist_ok=True)
    elif args.operation == "test":
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,
            # the following args are not used by test operation
            target_dir=None,
            hf_upload_path=None,
            private=False,
        )
    else:
        raise NotImplementedError(f"Unknown operation: {args.operation}")

    if config.backend == "fsdp":
        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()


if __name__ == "__main__":
    main()
