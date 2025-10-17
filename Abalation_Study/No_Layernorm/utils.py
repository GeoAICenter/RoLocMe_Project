import os
import cv2
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=0)
def crop(half_box_size, row, col, map_1):
    row_tl, col_tl = jnp.maximum(row - half_box_size, 0), jnp.maximum(col - half_box_size, 0)
    row_br, col_br = jnp.minimum(row + half_box_size, 255), jnp.minimum(col + half_box_size, 255)
    (
        missing_rows,
        missing_cols
    ) = (2 * half_box_size + 1) - (jnp.abs(row_br - row_tl) + 1), (2 * half_box_size + 1) - (
            jnp.abs(col_br - col_tl) + 1)
    row_tl, col_tl = jnp.maximum(row_tl - missing_rows, 0), jnp.maximum(col_tl - missing_cols, 0)
    return jax.lax.dynamic_slice(
                map_1,
                (row_tl.sum(), col_tl.sum()),
                (2 * half_box_size + 1, 2 * half_box_size + 1)
            )


@jax.jit
def min_max_normalized_heatmap_signal_with_building(img, building):
    min_ = jnp.min(img)
    building_ = building * min_
    img_ = img * building + building_
    max_ = jnp.max(img)
    return (img_ - min_) / (max_ - min_)


def get_all_maps(
        path_to_DPM_imgs,
        path_to_building_imgs,
        path_to_floodfill_data,
):
    dpms_imgs = os.listdir(path_to_DPM_imgs)
    dpms_list = []
    floodfill = []
    buildings_list = []
    for dpm in dpms_imgs:
        im = cv2.imread(
            f'{path_to_DPM_imgs}/{dpm}',
            0
        ) / 255
        dpms_list.append(im)

        building = cv2.imread(
            f'{path_to_building_imgs}/{dpm.split("_")[0]}.png',
            0
        ) / 255
        buildings_list.append(building)

        floodfill.append(np.load(f'{path_to_floodfill_data}/{dpm.split(".")[0]}.npy'))
        # c = np.load(f'{path_to_floodfill_data}/{dpm.split(".")[0]}.npy')
        # c = np.where(c == -1, 100, c)
        # c = np.where(c == 0, 255, c)
        # cv2.imshow('x', c.astype(np.uint8))
        # cv2.waitKey(0)

    dpms_list = np.array(dpms_list)
    buildings_list = np.array(buildings_list)
    floodfill = np.array(floodfill)
    return dpms_list, buildings_list, floodfill
