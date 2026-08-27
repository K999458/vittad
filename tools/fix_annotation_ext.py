"""Ensure every image file_name in a COCO annotation file ends with .png."""
import argparse
import json


def update_annotations(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for img in data['images']:
        if not img['file_name'].endswith('.png'):
            img['file_name'] = img['file_name'] + '.png'

    with open(output_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='path to the COCO annotation json')
    parser.add_argument('-o', '--output', default=None,
                        help='output path (defaults to in-place update)')
    args = parser.parse_args()
    update_annotations(args.input, args.output or args.input)
