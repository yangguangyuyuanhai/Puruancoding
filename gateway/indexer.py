"""
文件索引器 - ZIP 解析 + 目录扫描 + DreamSim 配对
负责将用户输入（目录或 ZIP）转化为可处理的图片路径列表
SAM3: 返回图片路径列表 [path1, path2, ...]
DreamSim: 返回图片对列表 [{"before": path, "after": path}, ...]
"""
from __future__ import annotations

import logging    # 日志
import os         # 文件系统操作
import zipfile    # ZIP 文件解压
import tempfile   # 临时目录（预留）
from pathlib import Path  # 路径处理（预留）

# 导入支持的图片扩展名集合
from shared.config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def scan_images_from_directory(directory: str) -> list[str]:
    """
    递归扫描目录下所有图片文件
    Args:
        directory: 要扫描的目录路径
    Returns:
        按路径排序的图片绝对路径列表
    """
    images = []
    # os.walk 递归遍历目录树
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            # 提取文件扩展名并转小写
            ext = os.path.splitext(f)[1].lower()
            # 只保留支持的图片格式
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    # 最终按完整路径排序，保证顺序确定性
    images.sort()
    logger.info(f"Scanned {len(images)} images from {directory}")
    return images


def extract_zip(zip_path: str, extract_to: str) -> str:
    """
    解压 ZIP 文件到指定目录
    Args:
        zip_path: ZIP 文件路径
        extract_to: 解压目标目录
    Returns:
        解压后的目录路径
    Raises:
        FileNotFoundError: ZIP 文件不存在
        ValueError: 非合法 ZIP 文件
    """
    # 验证文件存在性
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    # 验证是否为合法 ZIP
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    # 创建解压目标目录
    os.makedirs(extract_to, exist_ok=True)
    # 解压全部内容
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    logger.info(f"Extracted ZIP {zip_path} to {extract_to}")
    return extract_to


def index_sam3_input(input_path: str, extract_base: str) -> list[str]:
    """
    索引 SAM3 输入：支持目录或 ZIP 两种输入格式
    Args:
        input_path: 用户提供的输入路径（目录或 ZIP 文件）
        extract_base: ZIP 解压基础目录
    Returns:
        排序后的图片路径列表
    """
    if input_path.lower().endswith(".zip"):
        # ZIP 输入：先解压再扫描
        extract_dir = os.path.join(extract_base, "extracted")
        extract_zip(input_path, extract_dir)
        return scan_images_from_directory(extract_dir)
    elif os.path.isdir(input_path):
        # 目录输入：直接扫描
        return scan_images_from_directory(input_path)
    else:
        raise ValueError(f"Input path must be a directory or ZIP file: {input_path}")


def index_dreamsim_input(input_path: str, extract_base: str) -> list[dict]:
    """
    索引 DreamSim 输入：
    目录下需包含 before/ 和 after/ 子目录，同名文件自动配对
    Args:
        input_path: 包含 before/ 和 after/ 子目录的路径（或 ZIP）
        extract_base: ZIP 解压基础目录
    Returns:
        配对列表 [{"before": path, "after": path}, ...]
    """
    # 处理 ZIP 或目录输入
    if input_path.lower().endswith(".zip"):
        extract_dir = os.path.join(extract_base, "extracted")
        extract_zip(input_path, extract_dir)
        base_dir = extract_dir
    elif os.path.isdir(input_path):
        base_dir = input_path
    else:
        raise ValueError(f"Input path must be a directory or ZIP file: {input_path}")

    # 构造 before/ 和 after/ 目录路径
    before_dir = os.path.join(base_dir, "before")
    after_dir = os.path.join(base_dir, "after")

    # 验证两个子目录必须存在
    if not os.path.isdir(before_dir):
        raise ValueError(f"Missing 'before/' subdirectory in {base_dir}")
    if not os.path.isdir(after_dir):
        raise ValueError(f"Missing 'after/' subdirectory in {base_dir}")

    # 第一步：建立 before/ 文件的索引（文件名去扩展名 → 完整路径）
    before_files = {}
    for f in os.listdir(before_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            name = os.path.splitext(f)[0]  # 文件名去扩展名，用于配对匹配
            before_files[name] = os.path.join(before_dir, f)

    # 第二步：遍历 after/ 文件，按文件名与 before/ 配对
    pairs = []        # 成功配对的列表
    unmatched = []    # after/ 中找不到对应 before 的文件
    for f in sorted(os.listdir(after_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue  # 跳过非图片文件
        name = os.path.splitext(f)[0]
        after_path = os.path.join(after_dir, f)
        if name in before_files:
            # 配对成功：从 before_files 中 pop 出来（避免重复配对）
            pairs.append({
                "before": before_files.pop(name),
                "after": after_path,
            })
        else:
            # after/ 中的文件在 before/ 中没有对应
            unmatched.append(f)

    # 记录未配对的文件（不影响任务执行，仅记录警告）
    if unmatched:
        logger.warning(f"Unmatched after/ files (no before counterpart): {unmatched}")
    if before_files:
        logger.warning(f"Unmatched before/ files (no after counterpart): {list(before_files.keys())}")

    # 必须至少有一对配对成功
    if not pairs:
        raise ValueError("No matching image pairs found between before/ and after/ directories")

    # 按 after 路径排序，保证顺序确定性
    pairs.sort(key=lambda p: p["after"])
    logger.info(f"Indexed {len(pairs)} image pairs from {base_dir}")
    return pairs
