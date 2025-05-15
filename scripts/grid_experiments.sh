# Real directed networks
python run_experiments.py --exp real-networks-directed --recmethod "labelprop" 
python run_experiments.py --exp real-networks-directed --recmethod "gsignal" 
python run_experiments.py --exp real-networks-directed --recmethod "gnn" 

# Real undirected networks
python run_experiments.py --exp real-networks-undirected --recmethod "labelprop"
python run_experiments.py --exp real-networks-undirected --recmethod "gsignal" 
python run_experiments.py --exp real-networks-undirected --recmethod "gnn"

# Number of sensors
python run_experiments.py --exp real-sensors --recmethod "labelprop"
python run_experiments.py --exp real-sensors --recmethod "gnn" 
python run_experiments.py --exp real-sensors --recmethod "gsignal"
python run_experiments.py --exp synth-sensors --recmethod "gsignal" # Erdos + Freq,Sensors

# Network size
python run_experiments.py --exp network-size --recmethod "labelprop" # Erdos
python run_experiments.py --exp network-size2 --recmethod "labelprop" # Barabasi

# Real data
python run_experiments.py --exp real-data --recmethod "labelprop" 
python run_experiments.py --exp real-data --recmethod "gsignal"
python run_experiments.py --exp real-data --recmethod "gnn"