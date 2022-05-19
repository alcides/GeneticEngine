from __future__ import annotations

import sys
from abc import ABC
from abc import ABCMeta
from dataclasses import dataclass
from typing import Annotated
from typing import List
from typing import NamedTuple
from typing import Protocol

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from geneticengine.algorithms.gp.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import weight
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween

# from skimage.metrics import structural_similarity


class Root(ABC):
    pass


def load_image(path: str):
    img = Image.open(path)
    w, h = img.size
    img2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return {"img": img, "width": w, "height": h, "cv2": img2}


root_image = load_image(sys.args[0])  # load original image


@dataclass
class Polygon:
    x1: Annotated[int, IntRange[0, root_image["width"]]]
    y1: Annotated[int, IntRange[0, root_image["height"]]]
    x2: Annotated[int, IntRange[0, root_image["width"]]]
    y2: Annotated[int, IntRange[0, root_image["height"]]]
    x3: Annotated[int, IntRange[0, root_image["width"]]]
    y3: Annotated[int, IntRange[0, root_image["height"]]]
    red: Annotated[int, IntRange[0, 255]]
    green: Annotated[int, IntRange[0, 255]]
    blue: Annotated[int, IntRange[0, 255]]
    alpha: Annotated[int, IntRange[0, 255]]


@dataclass
class PolyImage(Root):
    polygons: Annotated[list[Polygon], ListSizeBetween(min=50, max=500)]

    def __str__(self):
        xs = ", ".join([str(p) for p in self.polygons])
        return f"[{xs}]"


def draw_image(im: PolyImage):
    image = Image.new("RGB", (root_image["width"], root_image["height"]))
    draw = ImageDraw.Draw(image, "RGBA")
    for p in im.polygons:
        ps = [
            (p.x1, p.y1),
            (p.x2, p.y2),
            (p.x3, p.y3),
        ]
        draw.polygon(ps, (p.red, p.green, p.blue, p.alpha))
    del draw
    return image


def fitness(im: PolyImage):
    polyimg = draw_image(im)
    icv2 = cv2.cvtColor(np.array(polyimg), cv2.COLOR_RGB2BGR)
    totalPixels = root_image["width"] * root_image["height"]
    return np.sum(
        (icv2.astype("float") - root_image["cv2"].astype("float")) ** 2,
    ) / float(totalPixels)
    # return structural_similarity(cv2, root_image["cv2"], multichannel=True)


class CB(Callback):
    def process_iteration(self, generation: int, population, time: float, gp):
        if generation % 10 == 0:
            bp = population[0].genotype
            polyimg = draw_image(bp)
            polyimg.save("out.png")


if __name__ == "__main__":
    g = extract_grammar([Polygon, PolyImage], Root)

    alg = GP(
        g,
        fitness,
        representation=treebased_representation,
        max_depth=20,
        population_size=200,
        number_of_generations=1000,
        probability_crossover=0.4,
        probability_mutation=0.1,
        minimize=True,
        n_elites=1,
        n_novelties=10,
        parallel_evaluation=True,
        callbacks=[CB()],
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    print("Fitness:", bf)
    polyimg = draw_image(bp)
    polyimg.save(
        "out.png",
    )
