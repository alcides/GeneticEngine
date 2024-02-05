## Optimizations

### Parallel Evaluation

On linux and macos, it is possible to perform evaluation in parallel, using multiple cores.

To enable such behaviour, you should replace the default SequentialEvaluator with ParallelEvaluator

```{eval-rst}
.. autoapiclass:: geneticengine.representations.tree.parallel_evaluation.ParallelEvaluator
```


## Sub-tree caching

A good way of implementing sub-tree caching is to use a fitness function (separate from methods) that uses [@lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache). Notice that your dataclasses need the `unsafe_hash` parameter.
