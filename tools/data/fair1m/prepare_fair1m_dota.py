import argparse
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert FAIR1M split with mismatched image/xml names to DOTA layout.')
    parser.add_argument(
        'src_split',
        help='FAIR1M split directory, e.g. dataset/FAIR1M1.0/train')
    parser.add_argument(
        'out_dir',
        help='Output DOTA-style split directory, e.g. dataset/FAIR1M1.0/dota/train')
    parser.add_argument(
        '--image-dir-name',
        default='images',
        help='Image subdirectory name under src_split.')
    parser.add_argument(
        '--xml-dir-name',
        default='labelXml',
        help='XML annotation subdirectory name under src_split.')
    parser.add_argument(
        '--copy-mode',
        choices=['copy', 'hardlink', 'symlink'],
        default='copy',
        help='How to place images into the output directory.')
    parser.add_argument(
        '--allow-missing-labels',
        action='store_true',
        help='Create image outputs even when the matching XML is missing.')
    return parser.parse_args()


def image_stem_to_xml_candidates(stem):
    candidates = [stem]
    if stem.startswith('P') and stem[1:].isdigit():
        candidates.append(str(int(stem[1:])))
    if stem.isdigit():
        candidates.append('P' + stem.zfill(4))

    seen = set()
    unique = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return [item + '.xml' for item in unique]


def find_xml(xml_dir, image_stem):
    for xml_name in image_stem_to_xml_candidates(image_stem):
        xml_path = osp.join(xml_dir, xml_name)
        if osp.exists(xml_path):
            return xml_path
    return None


def parse_fair1m_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for obj in root.findall('.//objects/object'):
        name_node = obj.find('./possibleresult/name')
        points = obj.findall('./points/point')
        if name_node is None or name_node.text is None or len(points) < 4:
            continue

        coords = []
        for point in points[:4]:
            if point.text is None:
                break
            xy = [part.strip() for part in point.text.split(',')]
            if len(xy) < 2:
                break
            coords.extend([float(xy[0]), float(xy[1])])
        else:
            cls_name = name_node.text.strip().replace(' ', '_')
            records.append((coords, cls_name))

    return records


def write_dota_txt(records, out_txt):
    with open(out_txt, 'w', encoding='utf-8') as f:
        for coords, cls_name in records:
            coord_text = ' '.join(f'{value:.2f}' for value in coords)
            f.write(f'{coord_text} {cls_name} 0\n')


def place_image(src, dst, mode):
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    elif mode == 'symlink':
        os.symlink(osp.abspath(src), dst)
    else:
        raise ValueError(f'Unsupported copy mode: {mode}')


def main():
    args = parse_args()
    image_dir = osp.join(args.src_split, args.image_dir_name)
    xml_dir = osp.join(args.src_split, args.xml_dir_name)
    out_image_dir = osp.join(args.out_dir, 'images')
    out_label_dir = osp.join(args.out_dir, 'labelTxt')

    if not osp.isdir(image_dir):
        raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
    if not osp.isdir(xml_dir) and not args.allow_missing_labels:
        raise FileNotFoundError(f'XML directory does not exist: {xml_dir}')

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    image_names = [
        name for name in os.listdir(image_dir)
        if osp.splitext(name)[1].lower() in IMG_EXTS
    ]
    image_names.sort()

    missing = []
    converted = 0
    for image_name in tqdm(image_names, desc='Preparing FAIR1M'):
        stem, ext = osp.splitext(image_name)
        src_image = osp.join(image_dir, image_name)
        out_image = osp.join(out_image_dir, stem + ext.lower())
        if not osp.exists(out_image):
            place_image(src_image, out_image, args.copy_mode)

        xml_path = find_xml(xml_dir, stem) if osp.isdir(xml_dir) else None
        if xml_path is None:
            missing.append(image_name)
            if args.allow_missing_labels:
                open(osp.join(out_label_dir, stem + '.txt'), 'w').close()
                continue
            continue

        records = parse_fair1m_xml(xml_path)
        write_dota_txt(records, osp.join(out_label_dir, stem + '.txt'))
        converted += 1

    print(f'Images processed: {len(image_names)}')
    print(f'Labels converted: {converted}')
    if missing:
        print(f'Missing labels: {len(missing)}')
        print('First missing examples: ' + ', '.join(missing[:10]))
        if not args.allow_missing_labels:
            raise RuntimeError(
                'Some images have no matching XML. Re-run with '
                '--allow-missing-labels only if this is expected.')
    print(f'DOTA-style output: {args.out_dir}')


if __name__ == '__main__':
    main()
