from nppaillier.paillier import generate_paillier_keypair
from communication.core import Communicate


class Config:

    @classmethod
    def init(cls,
             role,
             name,
             port,
             other,
             other_port
             ):
        cls.role = role
        cls.name = name
        cls.port = port
        cls.other = other
        cls.other_port = other_port
        return cls


pub, qub = generate_paillier_keypair()
Config.init(role='host',
            name="alice",
            port=7072,
            other='bob',
            other_port=8082)
com = Communicate()
