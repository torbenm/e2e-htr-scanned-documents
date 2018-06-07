
class Context(object):

    def __init__(self, contextObj):
        self.context = contextObj

    def __getattribute__(self, name):
        if name == "context":
            return self.__getattribute__(super, name)
        if name in self.context:
            return self.context[name]
        return None

ctx = Context({
    'exists': True
})

print ctx.exists
