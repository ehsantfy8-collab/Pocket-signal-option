# Minimal reimplementation of imghdr for compatibility
# Source: CPython standard library (simplified)

def what(file, h=None):
    if h is None:
        with open(file, 'rb') as f:
            h = f.read(32)

    for name, test in tests:
        res = test(h, file)
        if res:
            return res
    return None


def test_jpeg(h, f):
    if h[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'

def test_png(h, f):
    if h[:8] == b'\211PNG\r\n\032\n':
        return 'png'

def test_gif(h, f):
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'

def test_tiff(h, f):
    if h[:2] in (b'MM', b'II'):
        return 'tiff'

def test_bmp(h, f):
    if h[:2] == b'BM':
        return 'bmp'

def test_webp(h, f):
    if h[:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'


tests = [
    ('jpeg', test_jpeg),
    ('png', test_png),
    ('gif', test_gif),
    ('tiff', test_tiff),
    ('bmp', test_bmp),
    ('webp', test_webp),
]
