
class Config:
    compute_type = 'single'

    def __init__(self):
        pass

    @classmethod
    def init(cls, compute_type):
        cls.compute_type = compute_type
