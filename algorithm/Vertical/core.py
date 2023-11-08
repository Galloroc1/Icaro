class BaseModel:
    name = 'base'

    def train(self,**kwargs):
        raise

    def predict(self,**kwargs):
        raise


class BaseModelGuest(BaseModel):
    role = "Guest"

    def train(self):
        pass


class BaseModelHost(BaseModel):
    role = "Host"

    def train(self,**kwargs):
        pass


class SecurityModelGuest(BaseModelGuest):
    pass


class SecurityModelHost(BaseModelHost):
    pass


class Module:
    pass


