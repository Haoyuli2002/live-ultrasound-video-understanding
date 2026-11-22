"""CPU tests with real, tiny dense/MoE models; no downloads or training jobs.

Run from the repo: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import (
    AutoModelForImageTextToText, Qwen3_5Config, Qwen3_5MoeConfig,
    Qwen2VLImageProcessorFast,
    Qwen2VLConfig, Qwen2VLForConditionalGeneration, PreTrainedTokenizerFast,
)

from lavis.models.streamready.backbone_qwen3_5 import Qwen35Backbone, CLIP_MEAN, CLIP_STD
from lavis.models.streamready.backbone import build_backbone, backbone_dtype
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.streamready.fusion import apply_long_term_fusion, set_long_term_fusion_queries
from lavis.models.blip2_models.streamready import StreamReady
from lavis.models.blip2_models.blip2 import Blip2Base


def tiny_config(moe=False):
    text = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        pad_token_id=0, eos_token_id=2, bos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    if moe:
        text.update(num_experts=2, num_experts_per_tok=1,
                    moe_intermediate_size=32, shared_expert_intermediate_size=32)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                  out_hidden_size=32, patch_size=16, temporal_patch_size=2,
                  spatial_merge_size=2, num_position_embeddings=16)
    return (Qwen3_5MoeConfig if moe else Qwen3_5Config)(
        text_config=text, vision_config=vision,
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )


def tiny_backbone(moe=False):
    config = tiny_config(moe)
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
    processor = Qwen2VLImageProcessorFast(
        patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    with patch("lavis.models.streamready.backbone_qwen3_5.AutoConfig.from_pretrained", return_value=config), \
         patch("lavis.models.streamready.backbone_qwen3_5.AutoModelForImageTextToText.from_pretrained", return_value=model), \
         patch("lavis.models.streamready.backbone_qwen3_5.AutoImageProcessor.from_pretrained", return_value=processor), \
         patch("lavis.models.streamready.backbone_qwen3_5.AutoTokenizer.from_pretrained", return_value=tokenizer):
        return Qwen35Backbone("tiny-local-model", torch_dtype=torch.float32)


class Qwen35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dense_and_moe_prefix_gradients(self):
        for moe in (False, True):
            with self.subTest(moe=moe):
                backbone = tiny_backbone(moe)
                backbone.train()
                self.assertFalse(backbone.model.training)
                prefix = torch.randn(1, 3, backbone.llm_width, requires_grad=True)
                ids = torch.tensor([[5, 6, 7, 2]])
                embeds = torch.cat([prefix, backbone.embed_tokens(ids)], dim=1)
                labels = torch.tensor([[-100, -100, -100, 5, 6, 7, 2]])
                logits, loss = backbone.lm_forward(embeds, torch.ones(1, 7, dtype=torch.long), labels)
                self.assertEqual(logits.shape, (1, 7, 64))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(prefix.grad.norm().item(), 0)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in backbone.parameters()))

    def test_visual_preprocessing_matches_native_and_is_causal(self):
        backbone = tiny_backbone()
        rgb = torch.rand(2, 3, 3, 32, 32)  # B,C,T,H,W
        mean = torch.tensor(CLIP_MEAN)[None, :, None, None, None]
        std = torch.tensor(CLIP_STD)[None, :, None, None, None]
        pixels = (rgb - mean) / std
        features = backbone.encode_frames(pixels)
        self.assertEqual(features.shape, (2, 3, 1, 32))
        frames = rgb.permute(0, 2, 1, 3, 4).reshape(-1, 3, 32, 32)
        packed = backbone.image_processor(images=list(frames), do_resize=False, do_rescale=False, return_tensors="pt")
        with torch.no_grad():
            expected = torch.stack(backbone.model.get_image_features(**packed).pooler_output).reshape_as(features)
        torch.testing.assert_close(features, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(features[:, :1], backbone.encode_frames(pixels[:, :, :1]), atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            backbone.encode_frames(torch.zeros(1, 3, 1, 30, 32))
        backbone.drop_language_model()
        backbone.eval()
        torch.testing.assert_close(features, backbone.encode_frames(pixels))

    def test_native_generation_and_repeated_calls(self):
        for moe in (False, True):
            with self.subTest(moe=moe):
                backbone = tiny_backbone(moe)
                prefix = torch.randn(1, 4, 32)
                mask = torch.ones(1, 4, dtype=torch.long)
                opts = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False)
                first = backbone.lm_generate(prefix, mask, **opts)
                second = backbone.lm_generate(prefix, mask, **opts)
                self.assertEqual(first.shape, (1, 3))
                torch.testing.assert_close(first, second)
                beam = backbone.lm_generate(prefix, mask, num_beams=2, **opts)
                self.assertEqual(beam.shape, (1, 3))
                # Same prefix, cached native generation versus full recomputation.
                with torch.no_grad():
                    full = backbone.model.generate(inputs_embeds=prefix, attention_mask=mask,
                        use_cache=False, pad_token_id=0, eos_token_id=2, **opts)
                torch.testing.assert_close(first, full)

    def test_factory_and_precision(self):
        for model_type in ("qwen3_5", "qwen3_5_moe"):
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type=model_type)), \
                 patch("lavis.models.streamready.backbone_qwen3_5.Qwen35Backbone") as ctor:
                self.assertIs(build_backbone("local"), ctor.return_value)
                self.assertEqual(ctor.call_args.args[1], torch.bfloat16)
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type="qwen2_vl")), \
             patch("lavis.models.streamready.backbone.Qwen2VLBackbone") as ctor:
            self.assertIs(build_backbone("legacy"), ctor.return_value)
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type="qwen3_5_text")):
            with self.assertRaises(ValueError):
                build_backbone("text-only")
        self.assertEqual(backbone_dtype("bf16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            backbone_dtype("wrong")

    def test_real_local_checkpoint_loading(self):
        from tokenizers import Tokenizer, models
        for moe in (False, True):
            with self.subTest(moe=moe), tempfile.TemporaryDirectory() as tmp:
                source = tiny_backbone(moe)
                source.model.save_pretrained(tmp)
                source.image_processor.save_pretrained(tmp)
                raw = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}, unk_token="[UNK]"))
                PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]").save_pretrained(tmp)
                loaded = build_backbone(tmp, torch_dtype=torch.float32, attn_implementation="eager")
                self.assertEqual(loaded.vision_width, 32)
                self.assertEqual(loaded.llm_width, 32)
                pixels = torch.randn(1, 3, 1, 32, 32)
                torch.testing.assert_close(source.encode_frames(pixels), loaded.encode_frames(pixels))

    def test_batched_masked_generation(self):
        backbone = tiny_backbone()
        prefix = torch.randn(2, 7, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
        opts = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False)
        batched = backbone.lm_generate(prefix, mask, **opts)
        for i in range(2):
            compact = prefix[i:i+1, mask[i].bool()]
            single = backbone.lm_generate(compact, torch.ones(compact.shape[:2], dtype=torch.long), **opts)
            torch.testing.assert_close(batched[i], single[0])

    def test_qwen2_path_on_modern_transformers(self):
        from lavis.models.streamready.backbone import Qwen2VLBackbone
        config = Qwen2VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                bos_token_id=1, eos_token_id=2, pad_token_id=0,
                rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, embed_dim=32, hidden_size=32, num_heads=4,
                patch_size=14, temporal_patch_size=2, spatial_merge_size=2),
        )
        model = Qwen2VLForConditionalGeneration(config)
        with patch("lavis.models.streamready.backbone.Qwen2VLForConditionalGeneration.from_pretrained", return_value=model), \
             patch("lavis.models.streamready.backbone.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(pad_token="pad")):
            backbone = Qwen2VLBackbone("tiny-qwen2", torch_dtype=torch.float32)
        features = backbone.encode_frames(torch.randn(1, 3, 2, 28, 28))
        self.assertEqual(features.shape, (1, 2, 1, 32))
        prefix = torch.randn(1, 3, 32, requires_grad=True)
        _, loss = backbone.lm_forward(prefix, torch.ones(1, 3, dtype=torch.long), torch.tensor([[-100, 3, 4]]))
        loss.backward()
        self.assertGreater(prefix.grad.norm().item(), 0)
        backbone.drop_language_model()
        backbone.eval()
        self.assertEqual(backbone.encode_frames(torch.randn(1, 3, 1, 28, 28)).shape, (1, 1, 1, 32))

    def test_streamready_end_to_end_and_readiness(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        raw = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "what": 3, "yes": 4}, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
        backbone = tiny_backbone()
        backbone.tokenizer = tokenizer

        def make_qformer(num_query_token, vision_width, **kwargs):
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
                                num_attention_heads=4, intermediate_size=64,
                                hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
            config.encoder_width = vision_width
            config.add_cross_attention = True
            config.cross_attention_freq = 2
            config.query_length = num_query_token
            config.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
            return BertLMHeadModel(config), torch.nn.Parameter(torch.randn(1, num_query_token, 32) * .02)

        with patch("lavis.models.blip2_models.streamready.build_backbone", return_value=backbone), \
             patch.object(Blip2Base, "init_tokenizer", return_value=tokenizer), \
             patch.object(Blip2Base, "init_Qformer", side_effect=make_qformer):
            model = StreamReady(backbone_model="tiny", backbone_precision="fp32", num_query_token=3,
                qformer_encoder_width=24, frame_capacity=1, num_centroids=2, num_prototypes=1,
                topk_prototypes=1, topk_centroids=2, use_contextual_memory=False, train_stage="reasoning",
                gradient_checkpointing=True)
        samples = {"image": torch.randn(1, 3, 3, 32, 32), "text_input": ["what"], "text_output": ["yes"]}
        model.train()
        loss = model(samples)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.core.output_proj.weight.grad.norm().item(), 0)
        # Zero-initialised fusion deliberately blocks Short's CE gradient on
        # the first pass. Once its output projection changes, it must flow
        # through the checkpointed fusion layers into Short as well.
        self.assertIsNotNone(model.core.short_query_tokens.grad)
        self.assertGreater(model.core.long_query_tokens.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.zero_grad(set_to_none=True)
        fusion = model.core.long_qformer.bert.encoder.layer[0].long_term_fusion
        torch.nn.init.normal_(fusion.cross_attn.cross_attn.out_proj.weight, std=.01)
        model(samples)["loss"].backward()
        self.assertGreater(model.core.short_query_tokens.grad.norm().item(), 0)
        model.zero_grad(set_to_none=True)
        model.core.requires_grad_(True)
        model.train_stage = "readiness"
        model._apply_train_stage()
        loss = model(samples)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.readiness_parameters()))
        self.assertTrue(all(p.grad is None for p in model.reasoning_parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong-backbone.pth"
            state = {k: v for k, v in model.state_dict().items() if k.startswith("core.")}
            state["core.output_proj.weight"] = torch.zeros(48, 32)
            torch.save({"model": state}, path)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                model.load_checkpoint(str(path))

    def test_model_config_override(self):
        from lavis.common.config import Config
        args = SimpleNamespace(cfg_path="lavis/projects/streamready/train.yaml", options=[
            "model.model_type", "qwen3_5", "model.backbone_model", "Qwen/Qwen3.5-4B",
            "model.image_size", "256", "datasets.video_qa_train.vis_processor.train.image_size", "256",
        ])
        cfg = Config(args)
        self.assertEqual(cfg.model_cfg.model_type, "qwen3_5")
        self.assertEqual(cfg.model_cfg.backbone_precision, "bf16")
        self.assertEqual(cfg.model_cfg.qformer_encoder_width, 1408)

    def test_launchers_pass_consistent_backbone_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Capture argv only. Never execute torchrun or submit any job.
            stub = Path(tmp) / "torchrun"
            stub.write_text('#!/bin/sh\nprintf "SR_ARG=%s\\n" "$@"\n')
            stub.chmod(0o755)
            env = {**os.environ, "PATH": tmp + os.pathsep + os.environ["PATH"],
                   "NUM_GPUS": "1", "MODEL_TYPE": "qwen3_5", "BACKBONE": "/models/Qwen3.5-4B",
                   "STAGE": "reasoning"}
            for key in ("EXTRA", "RESUME", "INIT_FROM", "IMAGE_SIZE", "BACKBONE_PRECISION", "OUTPUT_DIR"):
                env.pop(key, None)
            scripts = [("run_scripts/train.sh", []),
                       ("run_scripts/proready_qa/test.sh", ["/models/run.pth"]),
                       ("run_scripts/benchmarks/eval.sh", ["/models/run.pth"])]
            for script, args in scripts:
                with self.subTest(script=script):
                    result = subprocess.run(["bash", script, *args], env=env, check=True, text=True, capture_output=True)
                    argv = [line[len("SR_ARG="):] for line in result.stdout.splitlines() if line.startswith("SR_ARG=")]
                    opts = argv[argv.index("--options") + 1:]
                    values = dict(zip(opts[::2], opts[1::2]))
                    self.assertEqual(values["model.model_type"], "qwen3_5")
                    self.assertEqual(values["model.backbone_precision"], "bf16")
                    self.assertEqual(values["model.image_size"], "256")
                    sizes = [v for k, v in values.items() if k.endswith(".image_size")]
                    self.assertTrue(all(v == "256" for v in sizes))
                    if script.endswith("train.sh"):
                        self.assertIn("Qwen3.5-4B", values["run.output_dir"])


class QFormerCompatibilityTests(unittest.TestCase):
    def test_pretrained_roundtrip_fusion_and_gradients(self):
        config = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64)
        config.encoder_width = 24
        config.add_cross_attention = True
        config.cross_attention_freq = 2
        config.query_length = 3
        model = BertLMHeadModel(config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = BertLMHeadModel.from_pretrained(tmp).eval()
        loaded.resize_token_embeddings(65)
        q = torch.randn(1, 4, 32, requires_grad=True)  # queries plus RDY
        visual = torch.randn(1, 5, 24, requires_grad=True)
        kwargs = dict(input_ids=torch.tensor([[1, 2]]), query_embeds=q,
                      attention_mask=torch.ones(1, 6), encoder_hidden_states=visual,
                      return_dict=True)
        before = loaded.bert(**kwargs).last_hidden_state
        apply_long_term_fusion(loaded.bert, num_heads=4)
        set_long_term_fusion_queries(loaded.bert, torch.randn(1, 3, 32))
        after = loaded.bert(**kwargs).last_hidden_state
        torch.testing.assert_close(before, after)  # zero-init preserves pretrained path
        (after * torch.randn_like(after)).sum().backward()
        self.assertGreater(q.grad.norm().item(), 0)
        self.assertGreater(visual.grad.norm().item(), 0)
        fusion = loaded.bert.encoder.layer[0].long_term_fusion
        self.assertGreater(fusion.cross_attn.cross_attn.out_proj.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
