from random import shuffle
from random import shuffle
import pandas as pd
from core.elements1 import *
import copy

#from pathlib import Path
#root = Path(__file__).parent
#folder = str(root) + '\\resources'
#file = str(folder) + '\\nodes.json'

network1 = Network('/Users/alessiopodesta/PycharmProjects/OVN-2022/resources/nodes.json')
network2 = Network('/Users/alessiopodesta/PycharmProjects/OVN-2022/resources/nodes.json', 'flex_rate')
network3 = Network('/Users/alessiopodesta/PycharmProjects/OVN-2022/resources/nodes.json', 'shannon')

node_labels = list(network1.nodes.keys())
connections = []
for i in range(100):
    shuffle(node_labels)
    connection = Connection(node_labels[0],node_labels[-1],1)
    connections.append(connection)

connections1 = copy.deepcopy(connections)
connections2 = copy.deepcopy(connections)
connections3 = copy.deepcopy(connections)

# fixed rate
streamed_connections1 = network1.stream(connections1, best='snr')
snrs1=[connection.snr for connection in streamed_connections1]
plt.hist(snrs1,bins=10)
plt.title('SNR Distribution fixed rate')
plt.xlabel('dB')
plt.show()

bit_rate1 = [connection.bit_rate for connection in streamed_connections1]
plt.hist(bit_rate1, bins=10)
plt.title('Average Bitrate Distribution fixed rate')
plt.xlabel('Gbps')
plt.show()

# flex rate
streamed_connections2 = network2.stream(connections2, best='snr')
snrs2=[connection.snr for connection in streamed_connections2]
plt.hist(snrs2,bins=10)
plt.title('SNR Distribution flex rate')
plt.xlabel('dB')
plt.show()

bit_rate2 = [connection.bit_rate for connection in streamed_connections2]
plt.hist(bit_rate2, bins=10)
plt.title('Average Bitrate Distribution flex rate')
plt.xlabel('Gbps')
plt.show()

# shannon
streamed_connections3 = network3.stream(connections3, best='snr')
snrs3=[connection.snr for connection in streamed_connections3]
plt.hist(snrs3,bins=10)
plt.title('SNR Distribution shannon')
plt.xlabel('dB')
plt.show()

bit_rate3 = [connection.bit_rate for connection in streamed_connections3]
plt.hist(bit_rate3, bins=10)
plt.title('Average Bitrate Distribution shannon')
plt.xlabel('Gbps')
plt.show()