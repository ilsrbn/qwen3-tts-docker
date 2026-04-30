import os
import uuid

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
ATTN_IMPL = os.getenv("QWEN_TTS_ATTN", "sdpa")

os.makedirs("outputs", exist_ok=True)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available inside container.")

print(f"Loading model: {MODEL_ID}")
print(f"Attention implementation: {ATTN_IMPL}")

model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=ATTN_IMPL,
)


def normalize_audio(wav):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            x = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            x = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)

    else:
        raise TypeError(f"Unsupported audio dtype: {x.dtype}")

    if x.ndim > 1:
        x = np.mean(x, axis=-1).astype(np.float32)

    return np.clip(x, -1.0, 1.0)


def generate(ref_audio, ref_text, target_text, language):
    if ref_audio is None:
        raise gr.Error("Upload or record reference audio first.")

    if not ref_text.strip():
        raise gr.Error(
            "Reference text is required. It should exactly match the reference audio."
        )

    if not target_text.strip():
        raise gr.Error("Target text is required.")

    sr, wav = ref_audio
    wav = normalize_audio(wav)

    wavs, out_sr = model.generate_voice_clone(
        text=target_text.strip(),
        language=language,
        ref_audio=(wav, int(sr)),
        ref_text=ref_text.strip(),
        max_new_tokens=2048,
    )

    output_path = f"outputs/qwen3_tts_{uuid.uuid4().hex[:8]}.wav"
    sf.write(output_path, wavs[0], out_sr)

    return output_path


with gr.Blocks(title="Local Qwen3-TTS Voice Clone") as demo:
    gr.Markdown("# Local Qwen3-TTS Voice Clone")
    gr.Markdown(
        "Record/upload your voice sample, paste the exact transcript, then generate new speech."
    )

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Reference audio",
                sources=["microphone", "upload"],
                type="numpy",
            )

            ref_text = gr.Textbox(
                label="Exact transcript of reference audio",
                lines=4,
                placeholder="Paste exactly what you said in the reference audio.",
            )

            language = gr.Dropdown(
                label="Target language",
                choices=[
                    "Auto",
                    "English",
                    "Russian",
                    "Chinese",
                    "Japanese",
                    "Korean",
                    "French",
                    "German",
                    "Spanish",
                    "Portuguese",
                    "Italian",
                ],
                value="English",
            )

        with gr.Column():
            target_text = gr.Textbox(
                label="Text to synthesize",
                lines=8,
                placeholder="Text that should be spoken with your cloned voice.",
            )

            btn = gr.Button("Generate", variant="primary")
            output = gr.Audio(label="Generated audio", type="filepath")

    btn.click(
        fn=generate,
        inputs=[ref_audio, ref_text, target_text, language],
        outputs=output,
    )


if __name__ == "__main__":
    ssl_certfile = os.getenv("QWEN_TTS_SSL_CERT")
    ssl_keyfile = os.getenv("QWEN_TTS_SSL_KEY")

    use_https = (
        ssl_certfile
        and ssl_keyfile
        and os.path.exists(ssl_certfile)
        and os.path.exists(ssl_keyfile)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssl_certfile=ssl_certfile if use_https else None,
        ssl_keyfile=ssl_keyfile if use_https else None,
        ssl_verify=False,
    )
