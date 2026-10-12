"""Small drawn markers for the policy map. Matplotlib has no refinery, pickaxe, or palm glyph."""

from matplotlib.path import Path


def _path(vertices, closed=False):
    codes = [Path.MOVETO]
    codes.extend([Path.LINETO] * (len(vertices) - 1))
    if closed:
        codes[-1] = Path.CLOSEPOLY
    return list(vertices), codes


def refinery():
    """Building with a stack. Used for oil and gas fields and LNG terminals."""
    building = [
        (-0.55, -0.45),
        (-0.55, 0.05),
        (0.35, 0.05),
        (0.35, -0.45),
        (-0.55, -0.45),
    ]
    stack = [
        (0.02, 0.05),
        (0.02, 0.48),
        (0.22, 0.48),
        (0.22, 0.05),
    ]
    roof = [
        (-0.62, 0.05),
        (-0.10, 0.32),
        (0.42, 0.05),
    ]
    vertices = []
    codes = []
    for part, close in ((building, True), (stack, True), (roof, False)):
        part_vertices, part_codes = _path(part, closed=close)
        vertices.extend(part_vertices)
        codes.extend(part_codes)
    return Path(vertices, codes)


def pickaxe():
    """Handle and head, placed at the centre of a mining polygon."""
    handle = [(-0.42, -0.48), (0.28, 0.32)]
    head_left = [(0.02, 0.12), (0.48, 0.42)]
    head_right = [(0.08, 0.42), (0.48, 0.08)]
    vertices = []
    codes = []
    for part in (handle, head_left, head_right):
        part_vertices, part_codes = _path(part, closed=False)
        vertices.extend(part_vertices)
        codes.extend(part_codes)
    return Path(vertices, codes)


def palm():
    """Trunk and three fronds."""
    trunk = [
        (-0.07, -0.52),
        (-0.07, 0.02),
        (0.07, 0.02),
        (0.07, -0.52),
        (-0.07, -0.52),
    ]
    frond_left = [(0.0, 0.02), (-0.42, 0.18), (-0.22, 0.42)]
    frond_up = [(0.0, 0.02), (0.0, 0.52)]
    frond_right = [(0.0, 0.02), (0.42, 0.18), (0.22, 0.42)]
    vertices = []
    codes = []
    for part, close in (
        (trunk, True),
        (frond_left, False),
        (frond_up, False),
        (frond_right, False),
    ):
        part_vertices, part_codes = _path(part, closed=close)
        vertices.extend(part_vertices)
        codes.extend(part_codes)
    return Path(vertices, codes)
