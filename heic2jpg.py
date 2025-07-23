from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

arg_parser = ArgumentParser(description="Convert HEIC files to JPG format.")
arg_parser.add_argument("-i", "--image", type=str, required=True, help="Path to the input image.")
arg_parser.add_argument("-o", "--output", type=str, required=False, help="Path to the output JPG image.", default=None)

args = arg_parser.parse_args()

input_image = Path(args.image)
if not input_image.exists():
    raise FileNotFoundError(f"The specified image file does not exist: {input_image}")

if args.output:
    output_image = Path(args.output)
else:
    output_image = Path(".").parent / input_image.with_suffix(".jpg")

if output_image.suffix.lower() != '.jpg':
    output_image = output_image.with_suffix('.jpg')

input_image = Image.open(args.image)

input_image.save(output_image, format='JPEG')