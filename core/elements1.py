import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import pandas as pd
from scipy import special as math
from scipy.constants import c
from scipy.constants import Planck
from scipy.constants import h

BER_t = 1e-3
Bn = 12.5e9  # noise bandwidth
Rs = 32e9  # lightpath symbol rate
df = 50e9  # lightpath frequency spacing betweer 2 channels


class Lightpath(object):
    def __init__(self, power, path, channel):
        self._sig_power = power
        self._path = path
        self._channel = channel
        self._noise_power = 0
        self._latency = 0
        self.Rs = 32.0e9
        self.df = 50.0e9

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
    def channel(self):
        return self._channel

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


class SignalInformation(Lightpath):

    def __init__(self, signal_power, path):
        super().__init__(signal_power, path, 0)
        self._signal_power = signal_power
        self._noise_power = 0
        self._latency = 0
        self._path = path
        self.Rs = 32.0e9
        self.df = 50.0e9

    def add_noise(self, value):
        self._noise_power += value

    def add_latency(self, value):
        self._latency += value

    def next(self):
        self._path = self._path[1:]

    @property
    def path(self):
        return self._path

    def getLatency(self):
        return self._latency

    def getNoisePower(self):
        return self._noise_power

    def getSignalPower(self):
        return self._signal_power


class Node(object):
    def __init__(self, node):
        self._label = node['label']
        self._position = node['position']  # tupla posizione matematica nello spazio (float, float)
        self._connected_nodes = node['connected_nodes']  # lista di nodi connessi [string]
        self._successive = {}  # dict di linee del nodo [Line]
        self._switching_matrix = None # matrice contenente i bitrate per connessione da nodo a nodo
        self._transceiver = ''

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, value):
        self._transceiver = value

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, value):
        self._switching_matrix = value

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, value):
        self._successive = value

    #  aggiorno il segnale modificando il suo percorso e chiamo
    #  il prossimo elemento da propagare
    def propagate(self, lightpath, occupation=False):
        path = lightpath.path  # signal information--> path
        if len(path) > 1:
            line_label = path[:2]  # the 1st and the 2nd element of path
            line = self.successive[line_label]
            lightpath.next()
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    #  aggiorno il segnale modificando il suo percorso e chiamo
    #  il prossimo elemento da propagare (lab4, questo lo fa senza occupare i canali)
    def probe(self, lightpath):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath = line.probe(lightpath)
        return lightpath


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self.successive = {}  # dict di nodi che succeddono alla linea [Node]
        self._state = ['free'] * 10
        self._n_amplifiers = int(self._length / 80e3)  # un amplificatore ogni 80km
        self._gain = 16
        self._noise_figure = 3  # usare 5 solo per es 9 lab8#
        self.alpha = 0.2e-3  # fiber loss dB/m
        self.beta = 2.3e-26  # usare 0.6e-26 solo per es7 lab8 #
        self.gamma = 1.27e-3

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @n_amplifiers.setter
    def n_amplifiers(self, value):
        self._n_amplifiers = value

    @property
    def state(self):
        return self._state

    def free_state(self):
        self._state = ['free'] * 10

    @state.setter
    def state(self, value):
        value = [v.strip().lower() for v in value]  # no spaces and all in lower c.
        if set(value).issubset({'free', 'occupied'}):
            self._state = value
        else:
            print('ERROR: No state value:', set(value) - {'free', 'occupied'})

    def latency_generation(self):
        latency = self._length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        # noise = signal_power / (2 * self.length) vecchia formula
        signal_power = lightpath.signal_power
        df = lightpath.df
        rs = lightpath.Rs
        noise = self.ase_generation() + self.nli_generation(signal_power, df, rs)
        return noise

    def propagate(self, lightpath, occupation=False):
        # Calcolo latenza e aggiungo
        latency = self.latency_generation()
        lightpath.add_latency(latency)
        # Calcolo eta e calcolo del rumore
        Nch = 10
        rs = lightpath.Rs
        Df = lightpath.df
        eta = 16 / (27 * np.pi) * np.log(
            np.pi ** 2 * self.beta * rs ** 2 * Nch ** (2 * rs / Df) / (2 * self.alpha)) * self.gamma ** 2 / (
                      4 * self.alpha * self.beta * rs ** 3)
        signal_power = self.optimized_launch_power(eta)
        lightpath.set_signal_power(signal_power)
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)
        # Verifico la disponibilità del canale e lo occupo
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state
        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)
        return lightpath

    # Come prima, senza occupare i canali
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

    def ase_generation(self):
        noise_fig = 10 ** (self._noise_figure / 10)
        gain_lin = 10 ** (self._gain / 10)
        f = 193.414e12
        Bn = 12.5e9  # GHz
        ASE = self.n_amplifiers * h * f * Bn * noise_fig * (gain_lin - 1)
        return ASE

    def nli_generation(self, signal_power, Df,  rs):
        Nch = 10 # per il momento, da implementare num_channels
        Pch = signal_power
        N_spans = self._n_amplifiers
        b = self.beta
        a = self.alpha / (20 * np.log10(np.e))
        eta = 16 / (27 * np.pi) * np.log(
            np.pi ** 2 * b * rs ** 2 * Nch ** (2 * rs / Df) / (2 * a)) * self.gamma ** 2 / (
                        4 * a * b * rs ** 3)
        nli_noise = N_spans * Bn * (Pch**3) * eta
        return nli_noise

    def optimized_launch_power(self, eta): # ricontrollare
        F = 10 ** (self.noise_figure / 10)
        G = 10 ** (self.gain / 10)
        f0 = 193.414e12
        olp = ((F * f0 * h * G) / (2 * eta)) ** (1 / 3)
        return olp


class Network(object):
    def __init__(self, json_path, transceiver='fixed_rate', num = 10):
        nodes_json = json.load(open(json_path, 'r'))
        self.nodes = {}
        self.lines = {}
        self._weighted_paths = None
        self._connected = False
        self._route_space = None  # dataframe che per ogni path descrive la disponibilità dei canali
        self._num_channels = num  # da implementare, per gestire num di canali diverso da standard 10

        for node_label in nodes_json:
            node_dict = nodes_json[node_label]
            node_dict['label'] = node_label
            # Creo l'istanza del nodo
            node = Node(node_dict)
            self.nodes[node_label] = node

            if 'transceiver' not in nodes_json[node_label].keys():
                # definisco la strategia leggendola o impostando lo standard
                node.transceiver = transceiver
            else:
                node.transceiver = nodes_json[node_label]['transceiver']

            # Creo le istanze delle linee
            for connect_n_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connect_n_label
                line_dict['label'] = line_label

                # Trovo le posizioni matematiche dei nodi
                node_position = np.array(nodes_json[node_label]['position'])
                connect_n_pos = np.array(nodes_json[connect_n_label]['position'])

                # Calcolo della distanza
                # length=sqrt((x1-x2)^2+(y1-y2)^2)
                line_dict['length'] = np.sqrt(np.sum(node_position - connect_n_pos) ** 2)

                line = Line(line_dict)
                self.lines[line_label] = line

    @property
    def route_space(self):
        return self._route_space

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def set_weighted_paths(self, s_power):
        if not self._connected:
            self.connect()
        node_labels = self.nodes.keys()
        all_couples = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label2 != label1:
                    all_couples.append(label1 + label2)

        # Creo il Dataframe e definisco la struttura
        columns = ['path', 'latency', 'noise', 'snr']
        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for couples in all_couples:
            for path in self.find_paths(couples[0], couples[1]):
                path_str = ''
                for node in path:
                    path_str += node + '-->'
                paths.append(path_str[:-3])

                # Ho costruito la prima parte del dataframe: per ogni coppia, guardo tutti i path possibili
                # e per ogni path, riscrivo ogni nodo come mi è richiesto dall'es ovvero con nodo->next nodo

                # ora propago
                s_i = SignalInformation(s_power, path)
                if couples in self.lines.keys():
                    line = self.lines[couples]

                    # calcolare eta ed effettuare l'optimized launch power
                    a = line.alpha / (20 * np.log10(np.e))
                    eta = 16 / (27 * np.pi) * np.log(
                        np.pi ** 2 * line.beta * Rs ** 2 * 10 ** (2 * Rs / df) / (2 * a)) * line.gamma ** 2 / (
                                  4 * a * line.beta * Rs ** 3)
                    s_power = line.optimized_launch_power(eta)
                s_i.set_signal_power(s_power)

                s_i = self.propagate(s_i, occupation=False)# dopo questo step avrò il signal coi valori finali
                # Non mi resta che salvare i valori e passare al prossimo percorso
                # ed una volta finito i percorsi, passare alla prossima coppia
                latencies.append(s_i.getLatency())
                noises.append(s_i.getNoisePower())
                snrs.append(10 * np.log10(s_i.getSignalPower() / s_i.getNoisePower()))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df

        # libero la route space
        route_space = pd.DataFrame()
        route_space['path'] = paths
        num = 10
        if self._num_channels:
            num = self._num_channels
        for i in range(num):  # standard 10
            route_space[str(i)] = ['free'] * len(paths)
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
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def free_space(self):
        states = ['free'] * len(self.route_space['path'])
        for l in self.lines.values():
            l.free_state()
        for i in range(10):
            self.route_space[str(i)] = states

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                       if ((key != label1) & (key != label2))]

        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}
        # Prendo tutte le chiavi escluse quelle dei nodi che mi sono stati dati in input
        # ovvero prendo tutti i possibili nodi di passaggio
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [
                    inner_path + cross_node
                    for cross_node in cross_nodes
                    if ((inner_path[-1] + cross_node in cross_lines) &
                        (cross_node not in inner_path))]

        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)

        return paths

    def find_best_snr(self, in_node, out_node):
        # all_paths = self.weighted_paths.path.values
        available_path = self.available_path(in_node, out_node)
        if available_path:
            inout_df = self.weighted_paths.loc[
                self.weighted_paths.path.isin(available_path)]
            best_snr = np.max(inout_df.snr.values)
            # best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
            best_path = inout_df.loc[ inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        # inout_paths = [path for path in all_paths
        #      if ((path[0] == input_node) and (path[-1] == output_node))]
        # inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(inout_paths)]
        # best_snr = np.max(inout_df.snr.values)
        # best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
        return best_path

    def find_best_latency(self, in_node, out_node):
        # all_paths = self.weighted_paths.path.values
        available_path = self.available_path(in_node, out_node)
        if available_path:
            # inout_paths = [path for path in all_paths
            #      if ((path[0] == input_node) and (path[-1] == output_node))]
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_path)]
            best_latency = np.min(inout_df.latency.values)
            # best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None
        return best_path

    # da ricontrollare funzionamento matrice
    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        switching_matrix = {}
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                inner_dict = {connected_node: np.zeros(10)}
                for connected_node2 in node.connected_nodes:
                    if connected_node2 != connected_node:
                        dict_tmp = {connected_node2: np.ones(10)}
                        inner_dict.update(dict_tmp)

                switching_matrix.update({connected_node: inner_dict})

                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
            node.switching_matrix = switching_matrix
            switching_matrix = {}

        self._connected = True

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start = self.nodes[path[0]]
        propagated_lightpath = start.propagate(lightpath, occupation)
        return propagated_lightpath

    # come sopra ma senza occupare
    def probe(self, lightpath):
        path = lightpath.path
        start = self.nodes[path[0]]
        propagated_lightpath = start.probe(lightpath)
        return propagated_lightpath

    #  per ogni elemento della connessione imposto l'attributo della latenza o del snr
    def stream(self, connections, best='latency'):
        streamed_connections = []
        for connection in connections:
            in_node = connection.in_node
            out_node = connection.out_node
            sig_power = connection.signal_power
            # self.set_weighted_paths(signal_power)
            if best == 'latency':
                path = self.find_best_latency(in_node, out_node)
            elif best == 'snr':
                path = self.find_best_snr(in_node, out_node)
            else:
                print('ERROR INPUT VALUE:', best)
                continue
            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                # prendo la disponibilità di tutti i canali di quel path
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]
                # Prendo il primo canale disponibile di quel path

                lightpath = Lightpath(sig_power, path, channel)
                rb = self.calculate_bit_rate(lightpath, self.nodes[in_node].transceiver)
                if rb == 0:
                    continue
                else:
                    connection.bit_rate = rb

                # ridondanza da ricontrollare
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]

                path = path.replace('-->', '')
                in_lightpath = Lightpath(sig_power, path, channel)
                out_lightpath = self.propagate(in_lightpath, True)
                connection.latency = out_lightpath.latency
                noise = out_lightpath.noise_power
                connection.snr = 10 * np.log10(in_lightpath.signal_power / noise)
                self.update_route_space(path, channel)
            else:
                connection.latency = 0
                connection.snr = 0
            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('-->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    @staticmethod
    def line_set_to_path(line_set):
        path = ""
        elements = list(itertools.permutations(list(line_set), len(list(line_set))))
        for i in range(len(elements)):
            flag = 1
            for j in range(len(elements[i]) - 1):
                if elements[i][j][1] != elements[i][j + 1][0]:
                    flag = 0
                j += 2
            if flag == 1:
                for j in range(len(elements[i])):
                    path += elements[i][j][0]
                return path

    def update_route_space(self, path, channel):
        all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = 'occupied'

                path_to_update = self.line_set_to_path(line_set)

                for j in range(len(path_to_update)):
                    if j not in (0, len(path_to_update) - 1):
                        if ((path_to_update[j - 1] in self.nodes[path_to_update[j]].connected_nodes) & (
                                path_to_update[j + 1] in self.nodes[path_to_update[j]].connected_nodes)):
                            self.nodes[path_to_update[j]].switching_matrix[path_to_update[j - 1]][path_to_update[j + 1]][channel] = 0

        self.route_space[str(channel)] = states

    def available_path(self, in_node, out_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1e-3)
        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == in_node) and (path[-1] == out_node))]
        available_path = []
        for path in all_paths:
            path_occupancy = self.route_space.loc[
                                 self.route_space.path == path].T.values[1:]
            if 'free' in path_occupancy:
                available_path.append(path)
            return available_path

    # def calculate_bit_rate(self, path, strategy):
    def calculate_bit_rate(self, lightpath, strategy):
        global BER_t
        Rs = lightpath.Rs
        global Bn
        path = lightpath.path
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

    def node_to_number(self, str):
        nodes = list(self.nodes.keys())
        nodes.sort()
        return nodes.index(str)

    def upgrade_traffic_matrix(self, mtx, nodeA, nodeB):
        A = self.node_to_number(nodeA)
        B = self.node_to_number(nodeB)
        connection = Connection(nodeA, nodeB, 1e-3)
        list_con = [connection]
        self.stream(list_con)
        btr = connection.bit_rate
        if btr == 0:
            mtx[A][B] = float('inf')
            return float('inf')
        mtx[A][B] -= btr
        return mtx[A][B]

class Connection(object):
    def __init__(self, in_node, out_node, s_power):
        self._in_node = in_node
        self._out_node = out_node
        self._sig_power = s_power
        self._latency = 0
        self._snr = 0
        self._rb = 0

    @property
    def bit_rate(self):
        return self._rb

    @bit_rate.setter
    def bit_rate(self, value):
        self._rb = value

    @property
    def in_node(self):
        return self._in_node

    @property
    def out_node(self):
        return self._out_node

    @property
    def signal_power(self):
        return self._sig_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, value):
        self._latency = value

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, value):
        self._snr = value
