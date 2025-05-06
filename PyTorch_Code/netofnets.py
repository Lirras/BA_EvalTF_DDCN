

class NetOfNets:
    list_of_nets = []

    def __init__(self):
        super().__init__()

    def add_network(self, network):
        self.list_of_nets.append(network)

    def evaluate_all(self):
        for item in self.list_of_nets:
            item.eval()

    def train_all(self):
        for item in self.list_of_nets:
            # todo: here forward nutzen mittels aller Netze
            item.apply()
