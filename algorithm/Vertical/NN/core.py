import torch

class MiddleModule:

    def forward(self,x):
        pass

    def backward(self):
        pass


class MiddleModuleGuest(MiddleModule):

    def __init__(self):
        pass


class MiddleModuleHost(MiddleModule):

    def __init__(self):
        pass


class VerticalNN:

    def __init__(self):
        pass

    def forward(self,x):
        pass

    def backward(self):
        pass
