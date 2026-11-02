"""Draw the Finomania icon (PNG for the app window, multi-size ICO for Windows)."""
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = 1024


def draw():
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    # vertical gradient rounded square
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    top, bot = (38, 99, 196), (13, 42, 104)
    for y in range(S):
        t = y / (S - 1)
        gd.line([(0, y), (S, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([40, 40, S - 40, S - 40], radius=210, fill=255)
    im.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(im)
    # grid lines
    for y in (330, 520, 710):
        d.line([(170, y), (S - 170, y)], fill=(255, 255, 255, 45), width=10)
    # volatility bars
    xs = [210, 300, 390, 480, 570, 660, 750]
    hs = [120, 190, 90, 240, 150, 300, 210]
    for x, h in zip(xs, hs):
        d.rounded_rectangle([x, 820 - h, x + 52, 820], radius=14, fill=(126, 182, 255, 150))
    # rising line
    pts = [(190, 700), (330, 560), (450, 620), (590, 420), (700, 470), (840, 250)]
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(pts, fill=(255, 255, 255, 120), width=70, joint="curve")
    im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(im)
    d.line(pts, fill=(255, 255, 255, 255), width=46, joint="curve")
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - 34, p[1] - 34, p[0] + 34, p[1] + 34], fill=(255, 255, 255, 255))
    d.ellipse([pts[-1][0] - 16, pts[-1][1] - 16, pts[-1][0] + 16, pts[-1][1] + 16], fill=(38, 99, 196, 255))
    return im


if __name__ == "__main__":
    im = draw()
    im.resize((256, 256), Image.LANCZOS).save(os.path.join(ROOT, "finomania", "gui", "assets", "finomania.png"))
    im.save(os.path.join(HERE, "finomania.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                                                        (256, 256)])
    im.resize((512, 512), Image.LANCZOS).save(os.path.join(ROOT, "docs", "icon.png"))
    print("icons written")
