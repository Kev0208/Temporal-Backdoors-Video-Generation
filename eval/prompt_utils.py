from __future__ import annotations


def decode_cli_text(text: str | None) -> str | None:
    if text is None or not text:
        return text
    if "\\" not in text:
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def build_triggered_caption(caption: str, trigger_text: str | None) -> str:
    caption = (caption or "").strip()
    trigger_text = (trigger_text or "").strip()
    if not trigger_text:
        return caption
    if not caption:
        return trigger_text
    return f"{trigger_text} {caption}"


def select_caption(row: dict, text_split: str, trigger_text: str | None = None) -> str:
    clean_caption = (row.get("caption_clean") or "").strip()
    if text_split == "clean":
        return clean_caption
    if trigger_text:
        return build_triggered_caption(clean_caption, trigger_text)
    triggered_caption = row.get("caption_triggered")
    if isinstance(triggered_caption, str) and triggered_caption.strip():
        return triggered_caption.strip()
    return clean_caption
