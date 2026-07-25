from collections.abc import MutableMapping

import numpy as np


REFERENCE_FRAME_SIZE = (2304, 1296)
REFERENCE_CIRCLE_CENTER = (1185, 732)
REFERENCE_CIRCLE_RADIUS = 215


def scale_count_region(size, reference_center=REFERENCE_CIRCLE_CENTER, reference_radius=REFERENCE_CIRCLE_RADIUS):
    base_w, base_h = REFERENCE_FRAME_SIZE
    width, height = size
    scale_x = width / base_w
    scale_y = height / base_h
    center = (
        int(round(reference_center[0] * scale_x)),
        int(round(reference_center[1] * scale_y)),
    )
    radius = int(round(reference_radius * min(scale_x, scale_y)))
    return center, radius


def get_count_region(size):
    return scale_count_region(size)


def point_in_circle(point, center, radius):
    return np.hypot(point[0] - center[0], point[1] - center[1]) <= radius


def point_circle_state(point, center, radius):
    return "inside" if point_in_circle(point, center, radius) else "outside"


def circle_transition(previous_state, current_state):
    if previous_state == "outside" and current_state == "inside":
        return "enter", 1
    if previous_state == "inside" and current_state == "outside":
        return "exit", -1
    return None, 0


def apply_signed_count_delta(region_counts: MutableMapping, class_id, delta):
    region_counts[class_id] += int(delta)
    return region_counts[class_id]
