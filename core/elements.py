import json
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import pandas as pd
from scipy import special as math

BER_t = 1e-3

Bn = 12.5e9  # noise bandwidth

Rs = 32e9  #lightpath symbol rate


class Lightpath(object):
    def __init__(self, power, path, channel):
        self._sig_power = power
        self._path = path
        self._latency = 0
        self._noise_power = 0
        self._channel = channel

    @property
    def signal_power(self):
        return self._sig_power

    def set_signal_power(self, value):
        self._sig_power = value

    @property
    def channel(self):
        return self._channel

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
        self._transceiver = ''

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

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, value):
        self._transceiver = value

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    def probe(self, lightpath):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath = line.probe(lightpath)
        return lightpath


class Line (object):
    def __init__(self, line_dict, num=10):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._state = ['free'] * num
        self._num_channels = num
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

    @property
    def num_channels(self):
        return self._num_channels

    @state.setter
    def set_state(self, state):
        #state = state.lower().strip()
        state = [s.lower().strip() for s in state]
        if set(state).issubset(set(['free', 'occupied'])):
            self._state = state
        else:
            print('Error: line states not recognized')

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

    def propagate (self, lightpath, occupation = False):
        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        signal_power = lightpath.signal_power
        noise = self.noise_generation(signal_power)
        lightpath.add_noise(noise)

        if occupation:
            # self._state = 'occupied'
            # ora che ho n canali e che quindi ho uno state per ogni canale devo aggiornare di conseguenza:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self._state = new_state

        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)
        return lightpath

    def probe(self, lightpath):
        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        signal_power = lightpath.signal_power
        noise = self.noise_generation(signal_power)
        lightpath.add_noise(noise)

        node = self.successive[lightpath.path[0]]
        lightpath = node.probe(lightpath)
        return lightpath


class Network(object):
    def __init__(self, json_path, num=10, transceiver='fixed_rate'):
        node_json = json.load(open(json_path, "r"))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None
        self._route_space = None  # sarà un df che per ogni path descrive la disponibilità dei canali
        self._num_channels = num

        for node_label in node_json:
            #Creo l'istanza del nodo
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            if 'transceiver' not in node_json[node_label].keys():
                # definisco la strategia leggendola o impostando uno standard
                node.transceiver = transceiver
            else:
                node.transceiver = node_json[node_label]['transceiver']

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

                line = Line(line_dict, self._num_channels)
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

    @property
    def route_space(self):
        return self._route_space

    def set_weighted_paths(self, signal_power):
        if not self._connected:
            self.connect()
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
                signal_information = SignalInformation(signal_power, path)
                signal_information = self.propagate(signal_information)  # dopo questo step avrò il signal coi valori finali
                # Non mi resta che salvare i valori e passare al prossimo percorso
                # ed una volta finito i percorsi, passare alla prossima coppia
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(
                    10 * np.log10(
                        signal_information.signal_power / signal_information.noise_power))  # formula data dal testo

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df

        route_space = pd.DataFrame()
        route_space['path'] = paths
        num = self._num_channels
        for i in range(num):
            route_space[str(i)] = ['free']*len(paths)
        self._route_space = route_space


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

    def propagate(self, lightpath, occupied = False):
        path = lightpath.path
        start = self.nodes[path[0]]
        propagated_lightpath = start.propagate(lightpath, occupied)
        return propagated_lightpath

    def probe(self, lightpath):
        path = lightpath.path
        start = self.nodes[path[0]]
        propagated_lightpath = start.probe(lightpath)
        return propagated_lightpath

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

        self._connected = True

    def available_paths(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) & (path[-1] == output_node))]
        #unavailable_lines = [line for line in self.lines if self.lines[line].state == 'occupied']
        available_paths = []
        for path in all_paths:
            #available = True
            path_occupancy = self.route_space.loc[self.route_space.path==path].T.values[1:]
            if 'free' in path_occupancy:
                available_paths.append(path)
            #for line in unavailable_lines:
             #   if line[0] + '->' + line[1] in path:
              #      available = False
               #     break
            #if available:
             #   available_paths.append(path)

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
            #best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
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
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
            #best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
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
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                # prendo la disponibilità di tutti i canali di quel path
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]
                # Prendo il primo canale disponibile di quel path

                lightpath = Lightpath(signal_power, path, channel)
                path1 = lightpath.path
                rb = self.calculate_bit_rate(path1, self.nodes[input_node].transceiver)
                if rb == 0:
                    continue
                else:
                    connection.bit_rate = rb

                path = path.replace('->', '')
                in_lightpath = Lightpath(signal_power, path, channel)
                out_lightpath = self.propagate(in_lightpath, True)
                connection.latency = out_lightpath.latency
                noise = out_lightpath.noise_power
                connection.snr = 10 * np.log10(signal_power / noise)
            else:
                connection.latency = 0
                connection.snr = 0
            streamed_connections.append(connection)

        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    def update_route_space(self, path, channel):
        all_paths = [self.path_to_line_set(p)
                     for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = 'occupied'
        self.route_space[str(channel)] = states

    # da qua comincia il casino
    def calculate_bit_rate(self, path, strategy):
        global BER_t
        # Rs = lightpath.Rs
        global Bn
        global Rs
        # path = lightpath.path
        Rb = 0
        GSNR_db = pd.array(self.weighted_paths.loc[self.weighted_paths['path'] == path]['snr'])[0]
        GSNR = 10 ** (GSNR_db / 10)

        if strategy == 'fixed_rate':
            if GSNR > 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                Rb = 100
            else:
                Rb = 0

        if strategy == 'flex_rate':
            if GSNR < 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                Rb = 0
            elif (GSNR > 2 * math.erfcinv(2 * BER_t) ** 2 * (Rs / Bn)) & (GSNR < (14 / 3) * math.erfcinv(
                    (3 / 2) * BER_t) ** 2 * (Rs / Bn)):
                Rb = 100
            elif (GSNR > (14 / 3) * math.erfcinv((3 / 2) * BER_t) ** 2 * (Rs / Bn)) & (GSNR < 10 * math.erfcinv(
                    (8 / 3) * BER_t) ** 2 * (Rs / Bn)):
                Rb = 200
            elif GSNR > 10 * math.erfcinv((8 / 3) * BER_t) ** 2 * (Rs / Bn):
                Rb = 400

        if strategy == 'shannon':
            Rb = 2 * Rs * np.log2(1 + Bn / Rs * GSNR) / 1e9

        return Rb


class Connection(object):
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0
        self._bit_rate = 0

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

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, value):
        self._bit_rate = value


