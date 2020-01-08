from numpy.random import seed

from protocols import EntanglementGeneration
from protocols import BBPSSW

import sequence
from sequence import topology
from sequence import timeline
from sequence import encoding
from sequence.process import Process
from sequence.entity import Entity
from sequence.event import Event


def three_node_test():
    tl = timeline.Timeline()

    # create nodes
    alice = topology.Node("alice", tl)
    bob = topology.Node("bob", tl)
    charlie = topology.Node("charlie", tl)
    nodes = [alice,bob]

    # create classical channels
    cc_ab = topology.ClassicalChannel("cc_ab", tl, distance=2e3, delay=2e5)
    cc_ac = topology.ClassicalChannel("cc_ac", tl, distance=1e3, delay=1e5)
    cc_bc = topology.ClassicalChannel("cc_bc", tl, distance=1e3, delay=1e5)
    # add ends
    cc_ab.set_ends([alice, bob])
    cc_ac.set_ends([alice, charlie])
    cc_bc.set_ends([bob, charlie])

    # create quantum channels
    qc_ac = topology.QuantumChannel("qc_ac", tl, distance=1e3)
    qc_bc = topology.QuantumChannel("qc_bc", tl, distance=1e3)

    # create memories
    NUM_MEMORIES = 10
    FIDELITY = 0.6
    MEMO_FREQ = int(1e6)
    memory_param_alice = {"fidelity": FIDELITY, "direct_receiver": qc_ac}
    memory_param_bob = {"fidelity": FIDELITY, "direct_receiver": qc_bc}
    alice_memo_array = topology.MemoryArray("alice_memory_array", tl,
                                            num_memories=NUM_MEMORIES,
                                            frequency=MEMO_FREQ,
                                            memory_params=memory_param_alice)
    bob_memo_array = topology.MemoryArray("bob_memory_array", tl,
                                          num_memories=NUM_MEMORIES,
                                          frequency=MEMO_FREQ,
                                          memory_params=memory_param_bob)
    alice.assign_component(alice_memo_array, "MemoryArray")
    bob.assign_component(bob_memo_array, "MemoryArray")
    qc_ac.set_sender(alice_memo_array)
    qc_bc.set_sender(bob_memo_array)

    # create BSM
    detectors = [{"efficiency": 1, "dark_count": 0, "time_resolution": 150, "count_rate": 25000000}] * 2
    bsm = topology.BSM("charlie_bsm", tl, encoding_type=encoding.single_atom, detectors=detectors)
    charlie.assign_component(bsm, "BSM")
    qc_ac.set_receiver(bsm)
    qc_bc.set_receiver(bsm)

    # assign quantum channels
    alice.assign_qchannel(qc_ac)
    bob.assign_qchannel(qc_bc)

    # create alice protocol stack
    egA = EntanglementGeneration(alice, middles=["charlie"], others=["bob"], fidelity=FIDELITY)
    bbpsswA = BBPSSW(alice, threshold=0.9)
    egA.upper_protocols.append(bbpsswA)
    bbpsswA.lower_protocols.append(egA)
    alice.protocols.append(egA)
    alice.protocols.append(bbpsswA)

    # create bob protocol stack
    egB = EntanglementGeneration(bob, middles=["charlie"], others=["alice"], fidelity=FIDELITY)
    bbpsswB = BBPSSW(bob, threshold=0.9)
    egB.upper_protocols.append(bbpsswB)
    bbpsswB.lower_protocols.append(egB)
    bob.protocols.append(egB)
    bob.protocols.append(bbpsswB)

    # create charlie protocol stack
    egC = EntanglementGeneration(charlie, others=["alice", "bob"])
    charlie.protocols.append(egC)

    # schedule events
    process = Process(egA, "start", [])
    event = Event(0, process)
    tl.schedule(event)

    # start simulation
    tl.init()
    tl.run()

    def print_memory(memoryArray):
        for i, memory in enumerate(memoryArray):
            print(i, memoryArray[i].entangled_memory, memory.fidelity)

    for node in nodes:
        memory = node.components['MemoryArray']
        print(node.name)
        print_memory(memory)


def multiple_node_test(n: int, runtime=1e12):
    # assert that we have an odd number of nodes
    assert n % 2 == 1, "number of nodes must be odd"

    tl = timeline.Timeline(runtime)

    # create nodes
    nodes = [None] * n
    for i in range(n):
        node = topology.Node("node_{}".format(i), tl)
        tl.entities.append(node)

        # end nodes
        if i % 2 == 1: 
            pass

        # middle nodes
        else:
            pass

    # schedule events

    # start simulation
    tl.init()
    tl.run()


if __name__ == "__main__":
    seed(1)
    three_node_test()


