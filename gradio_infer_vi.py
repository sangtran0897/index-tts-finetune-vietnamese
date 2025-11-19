#!/usr/bin/env python3
"""
Gradio interface for Vietnamese IndexTTS2 inference.

This wraps `indextts.infer_v2.IndexTTS2` and exposes the most common controls:
  - Vietnamese text prompt
  - Reference speaker audio (upload or microphone)
  - Optional emotion reference or emotion text
  - Sampling controls (top-k / top-p / temperature / beams)

Example usage:
    python3 gradio_infer_vi.py --device cuda:0 --fp16 --share
"""

from __future__ import annotations

import argparse
import tempfile
import time
import uuid
import atexit
from pathlib import Path
from typing import Dict, Optional

import gradio as gr
import torchaudio
from omegaconf import OmegaConf

from indextts.infer_v2 import IndexTTS2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio UI for Vietnamese IndexTTS2 inference.")
    parser.add_argument("--config", default="checkpoints/config.yaml", help="Base YAML config.")
    parser.add_argument(
        "--model-dir",
        default="checkpoints",
        help="Directory containing GPT/s2mel/vocoder assets referenced in the config.",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        default="checkpoints/gpt.pth",
        help="Fine-tuned GPT checkpoint to use.",
    )
    parser.add_argument(
        "--tokenizer",
        default="checkpoints/bpe.model",
        help="SentencePiece tokenizer path.",
    )
    parser.add_argument(
        "--default-speaker",
        default="ref_audio.wav",
        help="Fallback speaker prompt when the UI input is empty.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/vietnamese_from_metadata/gradio_outputs",
        help="Directory to store generated wav files.",
    )
    parser.add_argument("--device", default=None, help="Device string (e.g. cuda:0, cpu).")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference when running on CUDA.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing.")
    parser.add_argument("--theme", default="default", help="Gradio theme (default: default).")
    parser.add_argument(
        "--strip-punctuation",
        dest="strip_punctuation",
        action="store_true",
        default=False,
        help="Remove sentence-ending punctuation before synthesis (default: keep punctuation).",
    )
    return parser.parse_args()


class EngineWrapper:
    """Thin wrapper that keeps IndexTTS2 alive and cleans up temp config."""

    def __init__(
        self,
        *,
        config: Path,
        gpt_ckpt: Path,
        tokenizer: Path,
        model_dir: Path,
        device: Optional[str],
        fp16: bool,
        strip_sentence_punctuation: bool,
    ):
        cfg = OmegaConf.load(config)
        cfg.gpt_checkpoint = str(gpt_ckpt)
        dataset_cfg = cfg.get("dataset")
        if not dataset_cfg or "bpe_model" not in dataset_cfg:
            raise KeyError("Config missing dataset.bpe_model entry.")
        cfg.dataset["bpe_model"] = str(tokenizer)

        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        OmegaConf.save(cfg, tmp.name)
        tmp.close()
        self._cfg_path = Path(tmp.name)
        atexit.register(lambda: self._cfg_path.unlink(missing_ok=True))

        self._engine = IndexTTS2(
            cfg_path=str(self._cfg_path),
            model_dir=str(model_dir),
            device=device,
            use_fp16=fp16,
            strip_sentence_punctuation=strip_sentence_punctuation,
        )

    @property
    def engine(self) -> IndexTTS2:
        return self._engine


def resolve_path(path_str: str, description: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def build_generation_kwargs(top_k: float, top_p: float, temperature: float, num_beams: int) -> Dict[str, float]:
    kwargs: Dict[str, float] = {}
    if top_k and top_k > 0:
        kwargs["top_k"] = int(top_k)
    if top_p and top_p > 0:
        kwargs["top_p"] = float(top_p)
    if temperature and temperature > 0:
        kwargs["temperature"] = float(temperature)
    if num_beams and num_beams > 0:
        kwargs["num_beams"] = int(num_beams)
    return kwargs


def make_app(
    engine_wrapper: EngineWrapper,
    default_speaker: Path,
    output_dir: Path,
    theme: str,
    default_strip: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(
        text: str,
        speaker_prompt: Optional[str],
        emo_audio: Optional[str],
        emo_alpha: float,
        use_emo_text: bool,
        emo_text: str,
        interval_silence: int,
        max_text_tokens: int,
        top_k: float,
        top_p: float,
        temperature: float,
        num_beams: int,
        strip_punct: bool,
    ):
        if not text or not text.strip():
            return None, "❗️Please enter some text."

        speaker_path = Path(speaker_prompt) if speaker_prompt else default_speaker
        if not speaker_path.exists():
            return None, f"❗️Speaker audio not found: {speaker_path}"

        emo_audio_path = Path(emo_audio) if emo_audio else None
        if emo_audio_path and not emo_audio_path.exists():
            return None, f"❗️Emotion audio not found: {emo_audio_path}"

        generation_kwargs = build_generation_kwargs(top_k, top_p, temperature, num_beams)
        timestamp = int(time.time() * 1000)
        out_path = output_dir / f"vietts_{timestamp}_{uuid.uuid4().hex[:6]}.wav"

        try:
            engine_wrapper.engine.strip_sentence_punctuation = strip_punct
            engine_wrapper.engine.infer(
                spk_audio_prompt=str(speaker_path),
                text=text.strip(),
                output_path=str(out_path),
                emo_audio_prompt=str(emo_audio_path) if emo_audio_path else None,
                emo_alpha=emo_alpha,
                use_emo_text=use_emo_text,
                emo_text=emo_text or None,
                interval_silence=int(interval_silence),
                verbose=False,
                max_text_tokens_per_segment=int(max_text_tokens),
                **generation_kwargs,
            )
        except Exception as exc:  # pylint:disable=broad-except
            return None, f"❗️Inference failed: {exc}"

        if not out_path.exists():
            return None, "❗️Unexpected error: output file was not created."

        return str(out_path), f"✅ Saved to {out_path}"

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("## 🇻🇳 IndexTTS2 Vietnamese Inference")
        gr.Markdown(
            "Upload or record a speaker reference, type Vietnamese text, and optionally control emotion / sampling."
        )

        with gr.Row():
            text = gr.Textbox(label="Vietnamese Text", lines=4, placeholder="Nhập câu bạn muốn đọc...", value="")
        with gr.Row():
            speaker = gr.Audio(
                label="Speaker Reference (upload or record)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            emo_audio = gr.Audio(
                label="Emotion Reference (optional)",
                type="filepath",
                sources=["upload", "microphone"],
            )
        with gr.Row():
            emo_alpha = gr.Slider(label="Emotion Blend α", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
            use_emo_text = gr.Checkbox(label="Use Emotion Text", value=False)
            emo_text = gr.Textbox(label="Emotion Text (optional)", placeholder="ví dụ: vui vẻ, buồn bã...")

        with gr.Accordion("Sampling & Segmentation", open=False):
            with gr.Row():
                interval_slider = gr.Slider(
                    label="Silence Between Segments (ms)", minimum=0, maximum=500, step=50, value=200
                )
                max_tokens = gr.Slider(
                    label="Max Tokens per Segment", minimum=1200, maximum=1500, step=8, value=1200
                )
            gr.Markdown("Leave the following fields blank to fall back to config defaults (same as infer_v2.py).")
            with gr.Row():
                top_k_input = gr.Number(label="top_k", value=None, precision=0)
                top_p_input = gr.Number(label="top_p", value=None)
                temp_input = gr.Number(label="temperature", value=None)
                beams_input = gr.Number(label="num_beams", value=None, precision=0)

        run_button = gr.Button("Synthesize", variant="primary")
        strip_checkbox = gr.Checkbox(label="Strip sentence punctuation", value=default_strip)
        output_audio = gr.Audio(label="Generated Audio", type="filepath")
        status_box = gr.Textbox(label="Status", interactive=False)

        run_button.click(
            fn=synthesize,
            inputs=[
                text,
                speaker,
                emo_audio,
                emo_alpha,
                use_emo_text,
                emo_text,
                interval_slider,
                max_tokens,
                top_k_input,
                top_p_input,
                temp_input,
                beams_input,
                strip_checkbox,
            ],
            outputs=[output_audio, status_box],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    args = parse_args()

    config_path = resolve_path(args.config, "Config")
    gpt_ckpt = resolve_path(args.gpt_checkpoint, "GPT checkpoint")
    tokenizer_path = resolve_path(args.tokenizer, "Tokenizer")
    model_dir = resolve_path(args.model_dir, "Model directory")
    default_speaker = resolve_path(args.default_speaker, "Default speaker audio")
    output_dir = Path(args.output_dir).expanduser().resolve()

    engine_wrapper = EngineWrapper(
        config=config_path,
        gpt_ckpt=gpt_ckpt,
        tokenizer=tokenizer_path,
        model_dir=model_dir,
        device=args.device,
        fp16=args.fp16,
        strip_sentence_punctuation=args.strip_punctuation,
    )

    demo = make_app(
        engine_wrapper,
        default_speaker,
        output_dir,
        args.theme,
        args.strip_punctuation,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
