class Configuration:
    def __init__(self):
        self.input_size = 784
        self.hidden_size_1 = 200
        self.hidden_size_2 = 150
        self.hidden_size_3 = 100
        self.hidden_size_4 = 80

        self.output = 10
        self.batch_size = 200
        self.epoch = 100
        self.lr_rate = 0.01
