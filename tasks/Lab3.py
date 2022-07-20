from random import shuffle
from random import shuffle
import pandas as pd
from core.elements1 import *

#from pathlib import Path
#root = Path(__file__).parent
#folder = str(root) + '\\resources'
#file = str(folder) + '\\nodes.json'

network = Network('/Users/alessiopodesta/PycharmProjects/OVN-2022/resources/nodes.json')
network.connect()
node_labels = list(network.nodes.keys())
connections = []
for i in range(100):
    shuffle(node_labels)
    connection = Connection(node_labels[0],node_labels[-1],1)
    connections.append(connection)

streamed_connections = network.stream(connections ,best='snr')
snrs=[connection.snr for connection in streamed_connections]
plt.hist(snrs,bins=10)
plt.title('SNR Distribution')
plt.show()