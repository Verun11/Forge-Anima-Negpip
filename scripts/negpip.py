import re
import torch
import modules.scripts
import modules.shared as shared
from modules.script_callbacks import on_ui_settings
from backend.text_processing.anima_engine import AnimaTextProcessingEngine
from backend.text_processing import parsing
from backend.nn.anima import SelfCrossAttention
import gradio as gr

OPT_DEBUG = "negpip_debug"

# ── Singleton mask storage ──────────────────────────────────────────
_negpip_mask = None


def _log(msg):
    if getattr(shared.opts, OPT_DEBUG, False):
        print(f"[NegPiP] {msg}")


# ── Negative weight regex ────────────────────────────────────────────
# Matches all valid negative decimal forms: -1.0, -.1, -0.1, -1
_RE_NEG_WEIGHT = re.compile(r"\(\s*([^:]+?)\s*:\s*(-\s*\d*\.?\d+)\s*\)")


# ── Hook 1: Custom parse + tokenize_line ─────────────────────────────
# We pre-process the prompt to extract negative weights BEFORE the
# engine's parser sees it. This guarantees we capture all valid
# negative decimal forms regardless of how the engine parses.
def hook_anima_engine(engine: AnimaTextProcessingEngine):
    # Always re-hook to avoid stale closures after code reloads
    orig_tokenize_line = engine._negpip_orig_tokenize_line if hasattr(engine, "_negpip_orig_tokenize_line") else engine.tokenize_line
    engine._negpip_orig_tokenize_line = orig_tokenize_line

    def patched_tokenize_line(line):
        global _negpip_mask

        # Scan the raw prompt for negative weights BEFORE parsing
        neg_map = {}  # text → weight
        for m in _RE_NEG_WEIGHT.finditer(line):
            text_part = m.group(1).strip()
            weight = float(m.group(2).replace(" ", ""))
            if weight < 0:
                neg_map[text_part] = weight
                _log(f"Detected negative weight: '{text_part}' = {weight}")

        # Call the original tokenize_line
        chunks = orig_tokenize_line(line)

        if not neg_map:
            return chunks

        # Build the mask from T5 multipliers (only place with weights)
        for chunk in chunks:
            t5_mults = chunk.t5_multipliers
            has_neg = any(w < 0 for w in t5_mults)

            if has_neg:
                t5_w = torch.tensor(t5_mults)
                t5_abs = torch.abs(t5_w)

                mask = (t5_w == t5_abs).int()
                mask[mask == 0] = -1
                mask = mask.unsqueeze(0).unsqueeze(-1)

                if mask.shape[1] < 512:
                    mask = torch.nn.functional.pad(
                        mask, (0, 0, 0, 512 - mask.shape[1]), value=1.0
                    )
                _negpip_mask = mask.float()

                neg_count = (mask.squeeze() == -1).sum().item()
                _log(f"Mask built: {neg_count} negative tokens out of {mask.shape[1]}")

                # Make T5 multipliers absolute (Qwen is always 1.0)
                chunk.t5_multipliers = [abs(w) for w in chunk.t5_multipliers]

        return chunks

    engine.tokenize_line = patched_tokenize_line
    engine._negpip_hooked = True
    _log("Hooked tokenize_line")


# ── Hook 2: Cross-attention ──────────────────────────────────────────
def hook_cross_attention():
    # Always re-hook to avoid stale closures after code reloads
    orig_forward = SelfCrossAttention._negpip_orig_forward if hasattr(SelfCrossAttention, "_negpip_orig_forward") else SelfCrossAttention.forward
    SelfCrossAttention._negpip_orig_forward = orig_forward

    def patched_forward(
        self, x, context=None, rope_emb=None, transformer_options={}
    ):
        negpip_mask = _negpip_mask

        if negpip_mask is None or context is None or self.is_selfattn:
            return orig_forward(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)

        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)

        # mask [1, 512, 1] → [1, ..., 512, 1, 1] to align with V's N dim
        m = negpip_mask.to(v).unsqueeze(-1)
        while m.dim() < v.dim():
            m = m.unsqueeze(1)

        v = v * m

        return self.compute_attention(q, k, v, transformer_options=transformer_options)

    SelfCrossAttention.forward = patched_forward
    SelfCrossAttention._negpip_hooked = True
    _log("Hooked SelfCrossAttention.forward")


# ── Script ───────────────────────────────────────────────────────────
class Script(modules.scripts.Script):
    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            enabled = gr.Checkbox(label="Enable NegPiP", value=True)
        return [enabled]

    def process(self, p, enabled=True, *args):
        global _negpip_mask
        _negpip_mask = None

        if not enabled:
            return

        if not hasattr(p.sd_model, "text_processing_engine_anima"):
            return

        # Force re-encoding every generation so our tokenize_line
        # hook runs and captures the mask. Without this, WebUI
        # caches conditioning for identical prompts and skips
        # tokenize_line entirely.
        p.clear_prompt_cache()

        hook_anima_engine(p.sd_model.text_processing_engine_anima)
        hook_cross_attention()

    def postprocess(self, p, processed, enabled=True, *args):
        global _negpip_mask
        _negpip_mask = None


# ── Settings ─────────────────────────────────────────────────────────
def ext_on_ui_settings():
    shared.opts.add_option(OPT_DEBUG, shared.OptionInfo(False, "NegPiP Debug Logging", section=("negpip", "NegPiP")))

on_ui_settings(ext_on_ui_settings)