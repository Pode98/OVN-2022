import pandas as pd
from core.elements import *
#from pathlib import Path
#root = Path(__file__).parent
#folder = str(root) + '\\resources'
#file = str(folder) + '\\nodes.json'

network = Network('/Users/alessiopodesta/PycharmProjects/OVN-2022/resources/nodes.json')
network.connect()
# Creo tutte le possibili coppie di nodi
node_labels = network.nodes.keys()
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
    for path in network.find_paths(pair[0], pair[1]):
        path_string = ''
        for node in path:
            path_string += node + '->'
        paths.append(path_string[:-2])

        # Ho costruito la prima parte del dataframe: per ogni coppia, guardo tutti i path possibili
        # e per ogni path, riscrivo ogni nodo come mi è richiesto dall'es ovvero con nodo->next nodo

        # Ora propago
        signal_information = SignalInformation(1, path)  # power = 1
        signal_information = network.propagate(signal_information)  # dopo questo step avrò il signal coi valori finali
        # Non mi resta che salvare i valori e passare al prossimo percorso
        # ed una volta finito i percorsi, passare alla prossima coppia
        latencies.append(signal_information.latency)
        noises.append(signal_information.noise_power)
        snrs.append(
            10 * np.log10(signal_information.signal_power / signal_information.noise_power))  # formula data dal testo

df['paths'] = paths
df['latency'] = latencies
df['noise'] = noises
df['snr'] = snrs
