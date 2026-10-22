"""Translate merge failures at the ComfyUI node boundary."""
from __future__ import annotations

import logging
from pathlib import PureWindowsPath

import torch

from .core.errors import MergeError


MESSAGES = {
    "nonfinite_input": (
        "A LoRA factor, alpha, or blend coefficient contains NaN or Inf.", "LoRA因子・alpha・合成係数にNaN／Infがあります。",
        "Check the listed file/layer and strengths. Select a valid LoRA or turn OFF the affected slot.",
        "表示されたファイル・層と強度を確認し、正常なLoRAに変更するか該当枠をOFFにしてください。"),
    "nonfinite_output": (
        "The selected storage type produced NaN or Inf in the output.", "指定保存精度への変換後にNaN／Infが検出されました。",
        "Check the listed layer and strengths; select FP32 or BF16 when FP16 overflows. Nothing was saved.",
        "対象層と強度を確認し、FP16であふれる場合はFP32またはBF16を選んでください。ファイルは保存していません。"),
    "moss_mixed": (
        "A selected LoRA is not recognized as MOSS-Image.", "MOSS-Image用と認識されないLoRAが混在しています。",
        "Turn OFF the listed LoRA, or select only LoRAs for the same base model.", "表示されたLoRAをOFFにするか、同じベースモデル用のLoRAに変更してください。"),
    "model_mismatch": (
        "The LoRAs have different model families and no matching target layers.", "モデル系統と適用先レイヤーが一致していません。",
        "Use LoRAs for the same base model, or turn OFF the incompatible slots.", "同じベースモデル用のLoRAを選ぶか、互換性のない枠をOFFにしてください。"),
    "unsupported_adapter": (
        "The file contains an unsupported adapter format.", "未対応形式のアダプターが含まれています。",
        "Use a Standard LoRA with down/up factors, or turn OFF this slot.", "down/up因子を持つ標準LoRAに変更するか、この枠をOFFにしてください。"),
    "shape_mismatch": (
        "The target layers have incompatible input/output sizes or layouts.", "適用先レイヤーの入出力サイズまたは構造に互換性がありません。",
        "Use LoRAs for the same base model and matching layer sizes, or turn OFF an incompatible slot. Different LoRA ranks are allowed.",
        "同じベースモデル・互換なレイヤーサイズのLoRAに変更するか、不一致の枠をOFFにしてください。ランクが違うだけなら合成できます。"),
    "missing_targets": (
        "A target layer is missing from one or more selected LoRAs.", "一部のLoRAに存在しない適用先レイヤーがあります。",
        "Set Unmatched Layers to scaled_carry or drop, or use LoRAs sharing the same target layers.", "Unmatched Layersをscaled_carryまたはdropに変更するか、適用先が共通するLoRAを選んでください。"),
    "krea2_moss": (
        "Krea2 Gate-Aware cannot blend MOSS-Image LoRAs.", "MOSS-Image用LoRAではKrea2 Gate-Awareを使えません。",
        "Select Auto/Weighted for MOSS-Image, or replace the sources with Krea2 LoRAs.", "MOSS-ImageではAuto／Weightedを選ぶか、入力をKrea2用LoRAに変更してください。"),
    "krea2_count": (
        "Krea2 Gate-Aware requires exactly two enabled, selected LoRAs.", "Krea2 Gate-Awareには有効なLoRAがちょうど2本必要です。",
        "Enable two selected LoRA slots and turn OFF the others.", "使用するLoRAを2本だけONにし、それ以外をOFFにしてください。"),
    "no_sources": (
        "No enabled LoRA file is selected.", "ONになっているLoRAファイルが選択されていません。",
        "Select a LoRA file and turn its switch ON.", "LoRAファイルを選び、その枠のスイッチをONにしてください。"),
    "invalid_weights": (
        "The selected weights cannot be used with the current Weight Mode.", "現在の比率モードでは指定された重みを使用できません。",
        "Use a positive total in normalize mode. Select direct for negative weights; all-zero weights cannot be blended.",
        "normalizeでは重みの合計を正にしてください。負の重みはdirectを選びます。全て0では合成できません。"),
    "missing_file": (
        "The selected LoRA file is unavailable.", "選択されたLoRAファイルが見つかりません。",
        "Select an installed file, refresh the file list, or turn OFF this slot.", "存在するファイルを選び直すか、一覧を更新するか、この枠をOFFにしてください。"),
    "read_failed": (
        "The selected LoRA could not be read as a supported Standard LoRA.", "対応する標準LoRAとしてファイルを読み込めませんでした。",
        "Check the file format and completeness, select another Standard LoRA, or turn OFF this slot.", "形式・ファイルの破損や欠損を確認し、別の標準LoRAに変更するか、この枠をOFFにしてください。"),
    "save_failed": (
        "The blended LoRA could not be saved or verified.", "ブレンドLoRAの保存または保存内容の検証に失敗しました。",
        "Check Filename Prefix, output-folder permissions, and free disk space. See the detail for verification errors.",
        "保存名・保存先の書き込み権限・空き容量を確認してください。保存内容の検証エラーは下の詳細を確認してください。"),
    "out_of_memory": (
        "There is not enough memory to complete the blend.", "ブレンドに必要なメモリが不足しています。",
        "For CUDA errors, select Compute Device = cpu or auto_cpu fallback. Weighted uses less memory than the dense methods.",
        "CUDAの場合は計算デバイスをcpuにするか、auto_cpuフォールバックを選んでください。Weightedは密行列を使う方式よりメモリを抑えられます。"),
    "unexpected": (
        "The blend stopped with an unexpected error.", "予期しないエラーでブレンドが停止しました。",
        "Check the detail below and the selected files/settings. If it mentions CUDA, retry on CPU.",
        "下の詳細と選択ファイル・設定を確認してください。CUDAに関するエラーならCPUで再実行してください。"),
}


def describe_failure(error: Exception, slots: list[tuple[int, str]]) -> dict:
    if isinstance(error, MergeError):
        code, names, detail = error.code, error.sources, error.detail
    else:
        code = "out_of_memory" if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) else "unexpected"
        names, detail = [], str(error)
    en, ja, fix_en, fix_ja = MESSAGES.get(code, MESSAGES["unexpected"])
    sources = []
    for name in dict.fromkeys(names):
        matches = [(slot, selected) for slot, selected in slots if selected == name]
        if not matches:
            matches = [(slot, selected) for slot, selected in slots
                       if PureWindowsPath(selected).name == PureWindowsPath(name).name]
        if matches:
            sources.extend({"slot": slot, "name": selected} for slot, selected in matches)
        else:
            sources.append({"name": name})
    return {"code": code, "sources": sources, "reason": {"en": en, "ja": ja},
            "hint": {"en": fix_en, "ja": fix_ja}, "detail": detail}


def publish_failure(error: Exception, slots: list[tuple[int, str]]) -> None:
    # A fresh CPU-only test process has no ComfyUI server or executing-node context.
    try:
        from server import PromptServer
        from comfy_execution.utils import get_executing_context
    except ImportError:
        return
    try:
        context = get_executing_context()
        if context is None:
            return
        server = PromptServer.instance
        if server.client_id is None:
            return
        server.send_sync("lora_blend_one.error", {
            "node_id": str(context.node_id), "prompt_id": context.prompt_id,
            "error": describe_failure(error, slots),
        }, server.client_id)
    except Exception:
        # Reporting must not replace the original failure or turn it into a success.
        logging.debug("Could not deliver LoRA Blend One error status", exc_info=True)
