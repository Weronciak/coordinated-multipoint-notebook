import minorminer
from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
import dimod
import networkx as nx
import dwave.inspector


Q={('X[1]','X[1]',):-52,
('X[1]','X[2]',):16,
('X[1]','X[3]',):9,
('X[1]','X[4]',):17,
('X[1]','X[5]',):64,
('X[1]','X[6]',):32,
('X[1]','X[7]',):30,
('X[1]','X[8]',):14,
('X[2]','X[2]',):-96,
('X[2]','X[3]',):17,
('X[3]','X[3]',):-52,
('X[2]','X[4]',):33,
('X[3]','X[4]',):16,
('X[4]','X[4]',):-96,
('X[2]','X[5]',):126,
('X[3]','X[5]',):64,
('X[4]','X[5]',):126,
('X[5]','X[5]',):-189,
('X[2]','X[6]',):62,
('X[3]','X[6]',):30,
('X[4]','X[6]',):64,
('X[5]','X[6]',):256,
('X[6]','X[6]',):-157,
('X[2]','X[7]',):64,
('X[3]','X[7]',):32,
('X[4]','X[7]',):62,
('X[5]','X[7]',):256,
('X[6]','X[7]',):128,
('X[7]','X[7]',):-157,
('X[2]','X[8]',):32,
('X[3]','X[8]',):14,
('X[4]','X[8]',):32,
('X[5]','X[8]',):128,
('X[6]','X[8]',):64,
('X[7]','X[8]',):64,
('X[8]','X[8]',):-93,
}


solver = DWaveSampler(solver='Advantage2_prototype1.1', token='DEV-6d884649a230f40ade123f657681862752489a70')
solver_graph = solver.to_networkx_graph()

emb = minorminer.find_embedding(Q, solver_graph)
sampler = FixedEmbeddingComposite(solver, embedding=emb)


response = sampler.sample_qubo(Q, num_reads=2500, max_answers=100, annealing_time=168)
ener=list(response.data_vectors['energy'])
print(ener[0])


#print(ener)
print(str(list(response.samples())[0]))
print(emb)
print(response)
dwave.inspector.show(response)