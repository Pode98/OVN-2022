class SignalInformation(object):
    def __init__(self, power, path):
        self._sig_power = power
        self._path = path
        self._latency = 0
        self._noise_power = 0

    @property
    def signal_power(self):
        return self._sig_power

    def set_signal_power(self, value):
        self._sig_power = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, value):
        self._noise_power = value

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    def add_noise(self, value):
        self.noise_power += value

    def add_latency(self, value):
        self.latency += value


class Node(object):
    def __init__(self, node):
        self._label = node['label']
        self._position = node['position']
        self._connected_nodes = node['connected_nodes']
        self._successives = {}

    def label(self):
        return self._label

    def set_label(self, value):
        self._label = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @connected_nodes.setter
    def connected_nodes(self, value):
        self._connected_nodes = value


