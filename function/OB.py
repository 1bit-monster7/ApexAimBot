class Publisher_Ui:
    def __init__(self):
        self.subscribers = []

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)

    def notify(self, message):
        for subscriber in self.subscribers:
            subscriber.receive(message)


class Subscriber_Fun:
    def __init__(self, callback):
        self.callback = callback

    def receive(self, message):
        self.callback(message)
