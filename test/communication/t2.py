from communication.core import Federation

r = Federation(name="bob",role="host",other="alice",task_name="test")
r.init_connection()

r.receive(tag="test")
r.send("ok",tag="test2")
