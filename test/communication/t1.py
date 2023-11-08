from communication.core import Federation

r = Federation(name="alice",role="guest",other="bob",task_name="test")
r.init_connection()

r.send("hello",tag="test")
r.receive(tag="test2")
