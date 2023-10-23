import pika
from pickle import dumps as p_dumps
from pickle import loads as p_loads
import copy


class Federation:

    """
        I wrote a simple communication component, using pika, just for simplicity, maybe in the future
    """

    def __init__(self, name, role, other, task_name, host="localhost", port=5672):
        self.name = name
        self.role = role
        self.other = other
        self.task_name = task_name
        self.host = host
        self.port = port
        self.cached_msg_map = {}

        self._conn = None
        self._channel = None
        self._send_queue_name = f"name{name}sendto{other}task{task_name}"
        self._receive_queue_name = f"name{other}sendto{name}task{task_name}"

    def init_connection(self):
        params = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
            )
        self._conn = pika.BlockingConnection(params)
        if not self._channel:
            send_channel = self._conn.channel()
            send_channel.queue_declare(queue=self._send_queue_name, durable=False)

            receive_channel = self._conn.channel()
            receive_channel.queue_declare(queue=self._receive_queue_name, durable=False)
            self._channel = {
                "send": send_channel,
                "receive": receive_channel
            }

    def send(self, value, tag):
        print("send value :",value)
        self.__send_obj(value, tag)

    def __send_obj(self, value, tag: str) -> None:
        headers = {
            'is_stream': False
        }
        properties = pika.BasicProperties(
            content_type="text/plain",
            headers=headers,
            message_id=self.name,
            correlation_id=tag,
            delivery_mode=1,
        )
        self._channel['send'].basic_publish(body=p_dumps(value), properties=properties,
                                            exchange='',
                                            routing_key=self._send_queue_name,)

    def receive(
            self,
            tag: str,
    ) -> list:
        data = self.__receive_obj(self._channel['receive'], tag)
        print("receive data: ", data)
        return data

    def __receive_obj(self, channel, tag: str):
        data = self.get_data_from_cached(tag)
        if not (data is None):
            return data
        for method, properties, body in channel.consume(queue=self._receive_queue_name,):
            if not properties:
                raise ValueError(f'rabbitmq receive_obj timeout,tag<{tag}>,{self.task_name}')
            if properties.headers['is_stream']:
                continue
            data_name, data_tag = properties.message_id, properties.correlation_id
            channel.basic_ack(delivery_tag=method.delivery_tag)
            data = p_loads(body)
            if not (data_tag == tag):
                self.save_data_to_cached(data_tag, data)
                continue
            return data

    def get_data_from_cached(self, tag: str):
        '''
            从 self.cached_msg_map 缓存中去数据并且 del
        '''
        key = f'{self.other}-{tag}-{self.task_name}'
        data = self.cached_msg_map.get(key)
        if data is None:
            return
        data_cp = copy.deepcopy(data)
        del self.cached_msg_map[key]
        return data_cp

    def save_data_to_cached(
            self,
            tag: str,
            value,

    ) -> None:

        key = f'{self.name}-{tag}-{self.task_name}'
        self.cached_msg_map[key] = value
