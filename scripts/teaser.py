import argparse

from pdf2image import convert_from_path
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine multiple images into a teaser image')
    parser.add_argument('--input', type=str, help='Path to the input PDF files', nargs='+')
    args = parser.parse_args()

    w, h = 800, 800

    images = []
    for i, input in enumerate(args.input):
        image = Image.open(input)
        img_w, img_h = image.size
        if img_w < w and img_h < h:
            pos_x = (w - img_w) // 2
            pos_y = (h - img_h) // 2
            images.append((i * w + pos_x, pos_y, image))
        else:
            images.append((w * i, 0, image.resize((w, h))))

    total_width = w * len(images)
    teaser = Image.new('RGB', (w * len(images), h), (255, 255, 255))

    for x, y, img in images:
        teaser.paste(img, (x, y))

    teaser.convert('RGB').save('teaser.png')
