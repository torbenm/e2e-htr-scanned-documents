class Executable(object):

    def __call__(self, executor, epoch, session, graph):
        pass

    def extend_graph(self, graph):
        pass

    def continue(self, epoch: int) -> bool:
        return True

    def will_run(self, epoch: int) -> bool:
        return True
