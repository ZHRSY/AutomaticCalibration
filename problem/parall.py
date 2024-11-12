class JoblibParallelization:

    def __init__(self, aJoblibParallel, aJoblibDelayed, *args, **kwargs) -> None:
        super().__init__()
        self.parallel = aJoblibParallel
        self.delayed = aJoblibDelayed

    def __call__(self, f, X):
        return self.parallel(self.delayed(f)(x) for x in X)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("parallel", None)
        state.pop("delayed", None)
        return state
