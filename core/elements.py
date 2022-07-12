import json
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

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

    def next(self):
        self.path = self.path[1:]


class Node(object):
    def __init__(self, node):
        self._label = node['label']
        self._position = node['position']
        self._connected_nodes = node['connected_nodes']
        self._successive = {}

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

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, signal_information):
        path = signal_information.path
        if len(path)>1:
            line_lable = path[:2]
            line = self.successive[line_lable]
            signal_information.next()
            signal_information = line.propagate(signal_information)
        return signal_information



class Line (object):
    def __init__ (self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive ( self ):
        return self . _successive

    @successive.setter
    def successive (self , successive ):
        self . _successive = successive

    def latency_generation ( self ):
        latency = self . length / (c * 2 / 3)
        return latency

    def noise_generation (self , signal_power ):
        noise = signal_power / (2 * self . length )
        return noise

    def propagate (self, signal_information):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        # Update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information

class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, "r"))
        self._nodes = {}
        self._lines = {}
        for node_label in node_json:
            #Creo l'istanza del nodo
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            #Creo le istanze delle linee
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_lable = node_label + connected_node_label
                line_dict['label'] = line_lable

                #Trovo le posizioni matematiche dei nodi
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                #Calcolo della distanza
                length = np.sqrt(np.sum((node_position-connected_node_position)**2))
                line_dict['length'] = length

                line = Line(line_dict)
                self._lines[line_lable] = line

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    def draw(self):
        nodes = self.nodes
        for node_lable in nodes:
            n0 = nodes[node_lable]
            x0 = n0.position[0]
            y0 = n0.position[1]
            for connected_node in n0.connected_nodes:
                n1 = nodes[connected_node]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0,x1],[y0,y1],'b')
        plt.title('Network')
        plt.show()

    def propagate(self, signal_information):
        path = signal_information.path
        start = self.nodes[path[0]]
        propagated_signal_information = start.propagate(signal_information)
        return propagated_signal_information

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def find_paths(self, label1, label2):
        inner_paths = {}
        inner_paths['0'] = label1
        cross_lines = self.lines.keys()
        # Prendo tutte le chiavi escluse quelle dei nodi che mi sono stati dati in input
        # ovvero prendo tutti i possibili nodi di passaggio
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]

        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i+1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i+1)] += [inner_path + cross_node for cross_node in
                                          cross_nodes if ((inner_path[-1] + cross_node in cross_lines) &
                                                          (cross_node not in inner_path))]

        paths = []
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines: paths.append(path + label2)

        return paths


