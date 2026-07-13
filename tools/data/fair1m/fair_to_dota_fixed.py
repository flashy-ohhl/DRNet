#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAIR1M -> DOTA格式 转换脚本(修复版)

用法:
    python fair_to_dota_fixed.py <FAIR1M数据集根目录> <输出根目录>

示例:
    python fair_to_dota_fixed.py /data/zjj/PETDet/data/FAIR1M1.0 ./dataset

行为:
    - 会在 <输出根目录> 下生成一个与输入数据集同名的文件夹(如 FAIR1M1.0),
      结构为:
        <输出根目录>/FAIR1M1.0/images/P00000.png ...
        <输出根目录>/FAIR1M1.0/labelTxt/P00000.txt ...
    - 只递归查找名为 "images" 的目录下的图片文件(tif/tiff/jpg/png),
      不会误把 labelXml、zip、readme 等文件当成图片处理。
    - 图片按全局递增编号重命名(P00000.png 起),避免 part1/part2/test
      之间原始文件名重复导致互相覆盖。
    - 若同级存在 labelXml 目录且有同名 xml 标注文件,会转换为对应编号的
      labelTxt/*.txt(DOTA格式),编号与图片保持一致;若没有标注(如test集)
      则跳过,不会报错。
    - 图片读取失败(cv2.imread返回None)不会导致脚本崩溃,会记录到
      <输出根目录>/FAIR1M1.0/failed_images.txt 供后续排查,脚本会继续跑完。
"""

import os
import sys
import cv2
from tqdm import tqdm
from xml.dom.minidom import parse


IMG_EXTS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')


def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    box_list = []
    for obj in objects:
        name = obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points = obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox = []
        for point in points[:4]:
            x = point.childNodes[0].data.split(",")[0]
            y = point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name": name, "bbox": bbox})

    with open(tar, 'w') as file:
        print("imagesource:GoogleEarth", file=file)
        print("gsd:0.0", file=file)
        for box in box_list:
            ss = ""
            for f in box["bbox"]:
                ss += str(f) + " "
            name = box["name"].replace(" ", "_")
            ss += name + " 0"
            print(ss, file=file)


def collect_pairs(root_dir):
    """
    只查找名字叫 "images" 的目录,收集其中的图片文件;
    如果同级存在 labelXml 目录,尝试匹配同名xml标注文件。
    返回 [(img_path, label_path_or_None), ...]
    """
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) != "images":
            continue
        parent = os.path.dirname(dirpath)
        label_dir = os.path.join(parent, "labelXml")
        has_label_dir = os.path.isdir(label_dir)

        for f in sorted(filenames):
            if not f.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(dirpath, f)
            stem = os.path.splitext(f)[0]
            label_path = None
            if has_label_dir:
                candidate = os.path.join(label_dir, stem + ".xml")
                if os.path.exists(candidate):
                    label_path = candidate
            pairs.append((img_path, label_path))

    # 按路径排序,保证每次运行编号一致、可复现
    pairs.sort(key=lambda x: x[0])
    return pairs


def process_split(split_dir, out_split_path, split_label):
    """
    处理单个split(如train或test)目录下的所有images(可能横跨part1/part2等分片),
    合并输出到 out_split_path/images 和 out_split_path/labelTxt。
    """
    out_images = os.path.join(out_split_path, "images")
    os.makedirs(out_images, exist_ok=True)

    pairs = collect_pairs(split_dir)
    total = len(pairs)
    print(f"[{split_label}] 共找到 {total} 张图片,准备处理...")
    if total == 0:
        print(f"[{split_label}] 警告: 没有找到任何名为 'images' 目录下的图片文件。")
        return

    has_any_label = any(lp is not None for _, lp in pairs)
    out_labels = None
    if has_any_label:
        out_labels = os.path.join(out_split_path, "labelTxt")
        os.makedirs(out_labels, exist_ok=True)

    width = max(5, len(str(total)))  # 编号位数,至少5位(P00000)
    fail_log = []

    for idx, (img_path, label_path) in enumerate(tqdm(pairs, desc=f"processing {split_label} images")):
        img = cv2.imread(img_path, 1)
        if img is None:
            fail_log.append(img_path)
            continue

        out_name = f"P{idx:0{width}d}"
        out_img_path = os.path.join(out_images, out_name + ".png")
        ok = cv2.imwrite(out_img_path, img)
        if not ok:
            fail_log.append(img_path)
            continue

        if label_path is not None and out_labels is not None:
            out_label_path = os.path.join(out_labels, out_name + ".txt")
            try:
                solve_xml(label_path, out_label_path)
            except Exception as e:
                print(f"\n[{split_label}] 标注解析失败: {label_path}, 错误: {e}")

    if fail_log:
        fail_txt = os.path.join(out_split_path, "failed_images.txt")
        with open(fail_txt, "w") as f:
            f.write("\n".join(fail_log))
        print(f"[{split_label}] 共有 {len(fail_log)} 张图片读取/写入失败,详情见: {fail_txt}")
    else:
        print(f"[{split_label}] 全部图片处理成功,没有失败项。")


def fair_to_dota(in_path, out_root):
    in_path = os.path.normpath(in_path)
    dataset_name = os.path.basename(in_path)
    out_path = os.path.join(out_root, dataset_name)
    os.makedirs(out_path, exist_ok=True)

    # 优先按 train / test 这类顶层子目录分别处理,保留划分;
    # 若没有这种结构(比如直接传入了part1这一层),则整体当成一个split处理。
    top_entries = [d for d in sorted(os.listdir(in_path))
                    if os.path.isdir(os.path.join(in_path, d))]
    known_splits = [d for d in top_entries if d.lower() in ("train", "val", "test", "trainval")]

    if known_splits:
        for split in known_splits:
            split_dir = os.path.join(in_path, split)
            out_split_path = os.path.join(out_path, split)
            process_split(split_dir, out_split_path, split)
    else:
        process_split(in_path, out_path, dataset_name)

    print(f"完成。输出目录: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python fair_to_dota_fixed.py <FAIR1M数据集根目录> <输出根目录>")
        sys.exit(1)
    src = sys.argv[1]
    tar = sys.argv[2]
    fair_to_dota(src, tar)