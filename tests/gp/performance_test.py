



# @pytest.mark.parametrize(
#     "representation",
#     [
#         TreeBasedRepresentation,
#         GrammaticalEvolutionRepresentation,
#         StructuredGrammaticalEvolutionRepresentation,
#         DynamicStructuredGrammaticalEvolutionRepresentation,
#         StackBasedGGGPRepresentation,
#     ],
# )
# @pytest.mark.benchmark(group="mutation", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
# def test_bench_mutation(benchmark, representation):
#     r = NativeRandomSource(seed=1)
#     target_depth = 20
#     target_size = 100

#     repr = representation(grammar=grammar, max_depth=target_depth)

#     gs = GenericMutationStep()

#     def mutation():
#         p = SingleObjectiveProblem(lambda x: 3)
#         population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
#         for _ in range(100):
#             population = list(gs.iterate(p, SequentialEvaluator(), repr, r, population, target_size, 0))
#         return len(list(population))

#     n = benchmark(mutation)
#     assert n > 0


# @pytest.mark.parametrize(
#     "representation",
#     [
#         TreeBasedRepresentation,
#         GrammaticalEvolutionRepresentation,
#         StructuredGrammaticalEvolutionRepresentation,
#         DynamicStructuredGrammaticalEvolutionRepresentation,
#         StackBasedGGGPRepresentation,
#     ],
# )
# @pytest.mark.benchmark(group="crossover", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
# def test_bench_crossover(benchmark, representation):
#     r = NativeRandomSource(seed=1)
#     target_depth = 20
#     target_size = 100

#     repr = representation(grammar=grammar, max_depth=target_depth)

#     gs = GenericCrossoverStep()

#     def mutation():
#         p = SingleObjectiveProblem(lambda x: 3)
#         population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
#         for _ in range(100):
#             population = gs.iterate(p, SequentialEvaluator(), repr, r, population, target_size, 0)
#         return len(list(population))

#     n = benchmark(mutation)
#     assert n > 0
