class Config:
    ROLE = None
    NAME = None
    COM_MAP = None

    @classmethod
    def init(cls,
             role,
             name,
             com_map,

             ):
        cls.ROLE = role
        cls.NAME = name
        cls.COM_MAP = com_map
        return cls
