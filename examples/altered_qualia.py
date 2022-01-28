from dataclasses import dataclass
from random import random
from typing import Annotated, List
from PIL import Image, ImageChops, ImageDraw

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange


target_im = Image.open("examples/data/altered_qualia/darwin.png")
target_im_size = target_im.size

im = Image.new(target_im.mode, target_im_size)
draw = ImageDraw.Draw(im)
# draw.rectangle((100,100,20,20),outline=(150, 150, 150, 128))


@abstract
class Draw:
    def evaluate(self, **kwargs) -> float:
        ...

@dataclass
class DrawBlock(Draw):
    actions: Annotated[List[Draw], ListSizeBetween(2, 3)]
    
    def evaluate(self, **kwargs):
        for action in self.actions:
            action.evaluate()
    
    def __str__(self) -> str:
        return f"{self.actions}"
    
@dataclass
class Rectangle(Draw):
    x_fst_point: Annotated[int, IntRange(0,target_im_size[0])]
    y_fst_point: Annotated[int, IntRange(0,target_im_size[1])]
    x_sec_point: Annotated[int, IntRange(0,target_im_size[0])]
    y_sec_point: Annotated[int, IntRange(0,target_im_size[1])]
    colour_R: Annotated[int, IntRange(0,255)]
    colour_G: Annotated[int, IntRange(0,255)]
    colour_B: Annotated[int, IntRange(0,255)]
    colour_A: Annotated[float, FloatRange(0,1)]

    def evaluate(self, **kwargs):
        draw.rectangle(
            xy = (
                self.x_fst_point,
                self.y_fst_point,
                self.x_sec_point,
                self.y_sec_point
                ),
            outline=(
                self.colour_R,
                self.colour_G,
                self.colour_B,
                self.colour_A
                )
            )
    
    def __str__(self) -> str:
        return f"draw.rectangle(xy = ({self.x_fst_point},{self.y_fst_point},{self.x_sec_point},{self.y_sec_point}),outline=({self.colour_R},{self.colour_G},{self.colour_B},{self.colour_A}))"
    

def pixel_diff(im1, im2):
    """
    Calculates a black/white image containing all differences between the two input images.
    :param im1: input image A
    :param im2: input image B
    :return: a black/white image containing the differences between A and B
    """

    if im1.size != im2.size:
        raise Exception(f"different image sizes, can only compare same size images: A={im1.size} B={im2.size}")

    if im1.mode != im2.mode:
        raise Exception(f"different image mode, can only compare same mode images: A={im1.mode} B={im2.mode}")

    diff = (ImageChops.difference(im1, im2)).convert('L')
    s = sum(i * n for i, n in enumerate(diff.histogram()))

    return s

def fitness_function(individual : Draw, save = False):
    individual.evaluate()
    diff_to_target = pixel_diff(im,target_im)

    if save:
        im.save('bla', "PNG")
    
    return 1 / (diff_to_target + 1)


def preprocess():
    return extract_grammar([DrawBlock, Rectangle], Draw)

def evolve(g, seed, mode):
    alg = GP(
        g,
        fitness_function,
        representation=treebased_representation,
        number_of_generations=150,
        population_size=100,
        max_depth=15,
        favor_less_deep_trees=True,
        probability_crossover=0.75,
        probability_mutation=0.01,
        selection_method=("tournament", 2),
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        # safe_gen_to_csv=(f'{output_folder}/run_seed={seed}','all'),
    )
    (b, bf, bp) = alg.evolve(verbose=1)

    print("Best individual:", bp)
    print("Genetic Engine fitness:", fitness_function(bp,save=True))

    return b, bf

g = preprocess()

print(pixel_diff(im,target_im))
# print(fitness_function(im))
print(target_im_size)
print(target_im.mode)

evolve(g, 123, False)

# print(pixel_diff(target_im,target_im))
# print(fitness_function(target_im))
