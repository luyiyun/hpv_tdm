from time import perf_counter


class Timer:

    def __init__(self, desc=None) -> None:
        self.desc = desc

    def __enter__(self):
        self.t1 = perf_counter()
        return self

    def __exit__(self, a, b, c):
        self.eval = perf_counter() - self.t1
        if self.desc is not None:
            print("%s : %.4fs" % (self.desc, self.eval))
