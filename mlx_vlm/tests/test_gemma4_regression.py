import unittest
import mlx.core as mx
from mlx_vlm.utils import load_model
import tempfile
from pathlib import Path
from typing import Dict
import json

class TestGemma4Regression(unittest.TestCase):

    def test_load_mlx_format_with_extra_weights(self):
        """
        Verify that models in MLX format with extra legacy weights (like layer_scalar)
        can be loaded successfully by ensuring module-specific sanitization runs.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            text_config = {
                "model_type": "gemma4_text",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 8,
                "vocab_size": 32,
                "num_kv_shared_layers": 2,
                "hidden_size_per_layer_input": 0,
                "sliding_window": 32,
                "sliding_window_pattern": 2,
                "layer_types": ["full_attention"] * 4,
                "use_double_wide_mlp": False # prevent mlp size mismatch
            }
            config = {
                "model_type": "gemma4",
                "text_config": text_config,
                "vision_config": {
                    "model_type": "gemma4_vision",
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "default_output_length": 280,
                    "position_embedding_size": 10240,
                    "rms_norm_eps": 1e-6
                },
                "audio_config": None,
                "image_token_id": 31
            }
            with open(tmp_path / "config.json", "w") as f:
                json.dump(config, f)
            
            def layer_weights(i) -> Dict[str, mx.array]:
                return {
                    f"language_model.model.layers.{i}.self_attn.q_proj.weight": mx.zeros((16, 16)),
                    f"language_model.model.layers.{i}.self_attn.k_proj.weight": mx.zeros((8, 16)),
                    f"language_model.model.layers.{i}.self_attn.v_proj.weight": mx.zeros((8, 16)),
                    f"language_model.model.layers.{i}.self_attn.o_proj.weight": mx.zeros((16, 16)),
                    f"language_model.model.layers.{i}.self_attn.q_norm.weight": mx.zeros((8,)),
                    f"language_model.model.layers.{i}.self_attn.k_norm.weight": mx.zeros((8,)),
                    f"language_model.model.layers.{i}.input_layernorm.weight": mx.zeros((16,)),
                    f"language_model.model.layers.{i}.post_attention_layernorm.weight": mx.zeros((16,)),
                    f"language_model.model.layers.{i}.pre_feedforward_layernorm.weight": mx.zeros((16,)),
                    f"language_model.model.layers.{i}.post_feedforward_layernorm.weight": mx.zeros((16,)),
                    f"language_model.model.layers.{i}.mlp.gate_proj.weight": mx.zeros((32, 16)),
                    f"language_model.model.layers.{i}.mlp.up_proj.weight": mx.zeros((32, 16)),
                    f"language_model.model.layers.{i}.mlp.down_proj.weight": mx.zeros((16, 32)),
                    f"language_model.model.layers.{i}.layer_scalar": mx.ones((1,)),
                }

            weights = {
                "language_model.model.embed_tokens.weight": mx.zeros((32, 16)),
                "language_model.model.norm.weight": mx.zeros((16,)),
            }
            for i in range(4):
                weights.update(layer_weights(i))
            
            def vision_block_weights(i) -> Dict[str, mx.array]:
                return {
                    f"vision_tower.encoder.layers.{i}.self_attn.q_proj.linear.weight": mx.zeros((16, 16)),
                    f"vision_tower.encoder.layers.{i}.self_attn.k_proj.linear.weight": mx.zeros((16, 16)),
                    f"vision_tower.encoder.layers.{i}.self_attn.v_proj.linear.weight": mx.zeros((16, 16)),
                    f"vision_tower.encoder.layers.{i}.self_attn.o_proj.linear.weight": mx.zeros((16, 16)),
                    f"vision_tower.encoder.layers.{i}.self_attn.q_norm.weight": mx.zeros((8,)),
                    f"vision_tower.encoder.layers.{i}.self_attn.k_norm.weight": mx.zeros((8,)),
                    f"vision_tower.encoder.layers.{i}.mlp.gate_proj.linear.weight": mx.zeros((32, 16)),
                    f"vision_tower.encoder.layers.{i}.mlp.up_proj.linear.weight": mx.zeros((32, 16)),
                    f"vision_tower.encoder.layers.{i}.mlp.down_proj.linear.weight": mx.zeros((16, 32)),
                    f"vision_tower.encoder.layers.{i}.input_layernorm.weight": mx.zeros((16,)),
                    f"vision_tower.encoder.layers.{i}.post_attention_layernorm.weight": mx.zeros((16,)),
                    f"vision_tower.encoder.layers.{i}.pre_feedforward_layernorm.weight": mx.zeros((16,)),
                    f"vision_tower.encoder.layers.{i}.post_feedforward_layernorm.weight": mx.zeros((16,)),
                }

            weights.update({
                "vision_tower.patch_embedder.input_proj.weight": mx.zeros((16, 3 * 16**2)),
                "vision_tower.patch_embedder.position_embedding_table": mx.zeros((2, 10240, 16)),
                "embed_vision.embedding_projection.weight": mx.zeros((16, 16)),
            })
            for i in range(2):
                weights.update(vision_block_weights(i))

            # 1. Test the MLX-format path (where we filter by parameter keys)
            mx.save_safetensors(str(tmp_path / "model.safetensors"), weights, metadata={"format": "mlx"})
            model = load_model(tmp_path)
            self.assertIsNotNone(model)
            
            from mlx.utils import tree_flatten
            loaded_keys = set(dict(tree_flatten(model.parameters())).keys())
            self.assertNotIn(
                "language_model.model.layers.2.self_attn.k_proj.weight",
                loaded_keys,
                "Unused shared KV weight should have been filtered"
            )
            
            # 2. Test the non-MLX format path (where we run full sanitization)
            non_mlx_path = tmp_path / "non_mlx"
            non_mlx_path.mkdir()
            with open(non_mlx_path / "config.json", "w") as f:
                json.dump(config, f)
            # Do NOT remove legacy keys, test if non-MLX sanitization catches it
            mx.save_safetensors(str(non_mlx_path / "model.safetensors"), weights) # No metadata = non-MLX
            
            non_mlx_model = load_model(non_mlx_path)
            self.assertIsNotNone(non_mlx_model)
            
            non_mlx_keys = set(dict(tree_flatten(non_mlx_model.parameters())).keys())
            self.assertNotIn(
                "language_model.model.layers.2.self_attn.k_proj.weight",
                non_mlx_keys,
                "Unused shared KV weight should have been filtered from non-MLX checkpoint"
            )

if __name__ == "__main__":
    unittest.main()
