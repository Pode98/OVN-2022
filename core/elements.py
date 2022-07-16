import json
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import pandas as pd

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

    def propagate(self, signal_information, occupation = False):
        path = signal_information.path
        if len(path)>1:
            line_lable = path[:2]
            line = self.successive[line_lable]
            signal_information.next()
            signal_information = line.propagate(signal_information, occupation)
        return signal_information



class Line (object):
    def __init__ (self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._state = 'free'
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def state(self):
        return self._state

    @state.setter
    def set_state(self, state):
        state = state.lower().strip()
        if state in ['free', 'occupied']:
            self._state = state
        else:
            print('Error: line state not recognized')

    @property
    def successive ( self ):
        return self . _successive

    @successive.setter
    def set_successive (self , successive ):
        self . _successive = successive

    def latency_generation ( self ):
        latency = self . length / (c * 2 / 3)
        return latency

    def noise_generation (self , signal_power ):
        noise = signal_power / (2 * self . length )
        return noise

    def propagate (self, signal_information, occupation = False):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        # Update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        if occupation:
            self._state = 'occupied'

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information, occupation)
        return signal_information

class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, "r"))
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = None
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

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @weighted_paths.setter
    def set_weighted_paths(self, signal_power):  # Copiato da main precedente da rileggere per favore
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label2 != label1:
                    pairs.append(label1 + label2)

        # Creo il Dataframe e definisco la struttura
        columns = ['path', 'latency', 'noise', 'snr']
        df = pd.DataFrame()

        # Inizializzo vettori risultato
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Ho costruito la prima parte del dataframe: per ogni coppia, guardo tutti i path possibili
                # e per ogni path, riscrivo ogni nodo come mi è richiesto dall'es ovvero con nodo->next nodo

                # Ora propago
                signal_information = SignalInformation(1, path)  # power = 1
                signal_information = self.propagate(
                    signal_information)  # dopo questo step avrò il signal coi valori finali
                # Non mi resta che salvare i valori e passare al prossimo percorso
                # ed una volta finito i percorsi, passare alla prossima coppia
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(
                    10 * np.log10(
                        signal_information.signal_power / signal_information.noise_power))  # formula data dal testo

        df['paths'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self.weighted_paths = df


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

    def available_paths(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) & (path[-1] == output_node))]
        unavailable_lines = [line for line in self.lines if self.lines[line].state == 'occupied']
        available_paths = []
        for path in all_paths:
            available = True
            for line in unavailable_lines:
                if line[0] + '->' + line[1] in path:
                    available = False
                    break
            if available:
                available_paths.append(path)
        return available_paths

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

    def find_best_snr(self, input_node, output_node):
        #all_paths = self.weighted_paths.path.values
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
        else:
            best_path = None
        #inout_paths = [path for path in all_paths
         #      if ((path[0] == input_node) and (path[-1] == output_node))]
        #inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(inout_paths)]
        #best_snr = np.max(inout_df.snr.values)
        #best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
        return best_path

    def find_best_latency(self, input_node, output_node):
        #all_paths = self.weighted_paths.path.values
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
        #inout_paths = [path for path in all_paths
         #      if ((path[0] == input_node) and (path[-1] == output_node))]
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
        else:
            best_path = None
        return best_path

    def stream(self, connections, best='latency'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            #self.set_weighted_paths(signal_power)
            self.set_weighted_paths(1)
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: bestinput not recognized.Value:', best)
                continue
            if path:
                in_signal_information = SignalInformation(signal_power, path)
                out_signal_information = self.propagate(in_signal_information, True)
                connection.latency = out_signal_information.latency
                noise = out_signal_information.noise_power
                connection.snr = 10 * np.log10(signal_power / noise)
            else:
                connection.latency = None
                connection.snr = 0
            streamed_connections.append(connection)

        return streamed_connections


class Connection(object):
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr


