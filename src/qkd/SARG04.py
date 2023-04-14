"""Definition of SARG04 protocol implementation.

This module provides an implementation of the SARG04 protocol for quantum key distribution.
The SARG04 class must be attachedd to a node with suitable hardware, such as a QKDNode.
Also included in this module are a function to pair protocol instances (required before the start of transmission) and the message type used by the protocol.
"""

import math
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..topology.node import QKDNode

import numpy

from ..message import Message
from ..protocol import StackProtocol
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils import log


def pair_sarg04_protocols(sender: "SARG04", receiver: "SARG04") -> None:
    """Function to pair SARG04 protocol instances.

    Args:
        sender (SARG04): protocol instance sending qubits (Alice).
        receiver (SARG04): protocol instance receiving qubits (Bob).
    """

    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1

class SARG04MsgType(Enum):
    """Defines possible message types for ALG."""

    BEGIN_PHOTON_PULSE = auto()
    RECEIVED_QUBITS = auto()
    BASIS_LIST = auto()
    MATCHING_INDICES = auto()

class SARG04Message(Message):
    """Message used by SARG04 protocol.
        This message contains all information passed between SARG04 protocol instances.
    Messages of different types contain different information.

    Attributes:
        msg_type (SARG04MsgType): defines the message type.
        receiver (str): name of destination protocol instance.
        frequency (float): frequency for qubit generation (if `msg_type == BEGIN_PHOTON_PULSE`).
        light_time (float): lenght of time to send qubits (if `msg_type == BEGIN_PHOTON_PULSE`).
        start_time (int): simulation start time of qubit pulse (if `msg_type == BEGIN_PHOTON_PULSE`).
        wavelength (float): wavelength (in nm) of photons (if `msg_type == BEGIN_PHOTON_PULSE`).
        bases (List[int]): list of measurement bases (if `msg_type == BASIS_LIST`).
        indices (List[int]): list of indices for matching bases (if `msg_type == MATCHING_INDICES`).
    """

    def __init__(self, msg_type: SARG04MsgType, receiver: str, **kwargs):
        Message.__init__(self, msg_type, receiver)
        self.protocol_type = SARG04
        if self.msg_type is SARG04MsgType.BEGIN_PHOTON_PULSE:
            self.frequency = kwargs["frequency"]
            self.light_time = kwargs["light_time"]
            self.start_time = kwargs["start_time"]
            self.wavelength = kwargs["wavelength"]
        elif self.msg_type is SARG04MsgType.RECEIVED_QUBITS:
            pass
        elif self.msg_type is SARG04MsgType.BASIS_LIST:
            self.bases = kwargs["bases"]
            self.non_orthogonal = kwargs["non_orthogonal"]
        elif self.msg_type is SARG04MsgType.MATCHING_INDICES:
            self.indices = kwargs["indices"]
        else:
            raise Exception("SARG04 generated invalid message type {}".format(msg_type))

class SARG04(StackProtocol):
    """Implementation of SARG04 protocol.

    The SARG04 protocol uses photons to create a secure key between two QKD Nodes.

    Attributes:
        own (QKDNode): node that protocol instance is attached to.
        name (str): label for protocol instance.
        role (int): determines if instance is "alice" or "bob" node.
        working (bool): shows if protocol is currently working on a key.
        ready (bool): used by alice to show if protocol currently processing a generate_key request.
        light_time (float): time to use laser (in s).
        start_time (int): simulation start time of key generation.
        photon_delay (int): time delay of photon (ps).
        basis_lists (List[int]): list of bases that qubits are sent in.
        bit_lists (List[int]): list of 0/1 qubits sent (in bases from basis_lists).
        key (int): generated key as an integer.
        key_bits (List[int]): generated key as a list of 0/1 bits.
        another (SARG04): other SARG04 protocol instance (on opposite node).
        key_lengths (List[int]): list of desired key lengths.
        self.keys_left_list (List[int]): list of desired number of keys.
        self.end_run_times (List[int]): simulation time for end of each request.
    """
    def __init__(self, own: "QKDNode", name: str, lightsource: str, qsdetector: str, role=-1):
        """Constructor for SARG04 class.

        Args:
            own (QKDNode): node hosting protocol instance.
            name (str): name of protocol instance.
            lightsource (str): name of lightsource for QKD
            qsdetector (str): name of QSDetector for QKD

        Keyword Args:
            role (int): 0/1 role defining Alice and Bob protocols (default -1)
        """

        if own is None: # for unit testing
            return
        super().__init__(own, name)
        self.ls_name = lightsource
        self.qsd_name = qsdetector
        self.role = role

        self.working = False
        self.ready = True  # (for Alice) not currently processing a generate_key request
        self.light_time = 0  # time to use laser (measured in s)
        self.ls_freq = 0  # frequency of light source
        self.start_time = 0  # start time of light pulse
        self.photon_delay = 0  # time delay of photon (including dispersion) (ps)
        self.basis_lists = None
        self.bit_lists = None
        self.key = 0  # key as int
        self.key_bits = None  # key as list of bits
        self.another = None
        self.key_lengths = []  # desired key lengths (from parent)
        self.keys_left_list = []
        self.end_run_times = []

        # metrics
        self.latency = 0  # measured in seconds
        self.last_key_time = 0
        self.throughputs = []  # measured in bits/sec
        self.error_rates = []

        """               alice announces
                       | +x,+z | +x,-z | -x,+z | -x,-z
                  x,+1 |   -   |  -    |  +z   |  -z   
        Bob Sees  x,-1 |   +x  |  -z   |  -    |   -   
                  z,+1 |   -   |  +x   |  -    |  -x   
                  z,-1 |   +x  |  -    |  -x   |   - 

        z = 1, x = 0
        + = 0, - = 1

        We precompute the potential deductions that Bob could make
        as the set of potential cases and outcomes is entirely deterministic
        """
        self.lookup_table = [
            [None, None, (0,1), (1,1)],
            [(0,1), (1,1), None, None],
            [None, (0,0), None, (1,0)],
            [(0,0), None, (1,0), None]
        ]

    def pop(self, detector_index: int, time: int) -> None:
        """Method to receive detection events (currently unused)."""
        assert 0

    def push(self, length: int, key_num: int, run_time=math.inf) -> None:
        """Method to receive requests for key generation.

        Args:
            length (int): length of key to generate.
            key_num (int): number of keys to generate.
            run_time (int): max simulation time allowed for key generation (default inf).

        Side Effects:
            Will potentially invoke `start_protocol` method to start operations.
        """

        if self.role != 0:
            raise AssertionError("generate key must be called from Alice")

        log.logger.info(self.name + " generating keys, keylen={}, keynum={}".format(length, key_num))

        self.key_lengths.append(length)
        self.another.key_lengths.append(length)
        self.keys_left_list.append(key_num)
        end_run_time = run_time + self.own.timeline.now()
        self.end_run_times.append(end_run_time)
        self.another.end_run_times.append(end_run_time)

        if self.ready:
            self.ready = False
            self.working = True
            self.another.working = True
            self.start_protocol()


    def start_protocol(self) -> None:
        """Method to start protocol.

        When called, this method will begin the process of key generation.
        Parameters for hardware will be calculated, and a `begin_photon_pulse` method scheduled.

        Side Effects:
            Will schedule future `begin_photon_pulse` event.
            Will send a BEGIN_PHOTON_PULSE method to other protocol instance.
        """

        log.logger.debug(self.name + " starting protocol")

        if len(self.key_lengths) > 0:
            # reset buffers for self and another
            self.basis_lists = []
            self.another.basis_lists = []
            self.bit_lists = []
            self.another.bit_lists = []
            self.key_bits = []
            self.another.key_bits = []
            self.latency = 0
            self.another.latency = 0

            self.working = True
            self.another.working = True

            ls = self.own.components[self.ls_name]
            self.ls_freq = ls.frequency

            # calculate light time based on key length
            self.light_time = self.key_lengths[0] / (self.ls_freq * ls.mean_photon_num)

            # send message that photon pulse is beginning, then send bits
            self.start_time = int(self.own.timeline.now()) + round(self.own.cchannels[self.another.own.name].delay)
            message = SARG04Message(SARG04MsgType.BEGIN_PHOTON_PULSE, self.another.name,
                                  frequency=self.ls_freq, light_time=self.light_time,
                                  start_time=self.start_time, wavelength=ls.wavelength)
            self.own.send_message(self.another.own.name, message)

            process = Process(self, "begin_photon_pulse", [])
            event = Event(self.start_time, process)
            self.own.timeline.schedule(event)

            self.last_key_time = self.own.timeline.now()
        else:
            self.ready = True

    def begin_photon_pulse(self) -> None:
        """Method to begin sending photons.

        Will calculate qubit parameters and invoke lightsource emit method.
        Also records bits sent for future processing.

        Side Effects:
            Will set destination of photons for local node.
            Will invoke emit method of node lightsource.
            Will schedule another `begin_photon_pulse` event after the emit period.
        """
        
        log.logger.debug(self.name + " starting photon pulse")
        
        if self.working and self.own.timeline.now() < self.end_run_times[0]:
            self.own.destination = self.another.own.name

            # generate basis/bit list
            num_pulses = round(self.light_time * self.ls_freq)
            basis_list = numpy.random.choice([0, 1], num_pulses)
            bit_list = numpy.random.choice([0, 1], num_pulses)

            # control hardware
            lightsource = self.own.components[self.ls_name]
            encoding_type = lightsource.encoding_type
            state_list = []
            for i, bit in enumerate(bit_list):
                state = (encoding_type["bases"][basis_list[i]])[bit]
                state_list.append(state)
            lightsource.emit(state_list)

            self.basis_lists.append(basis_list)
            self.bit_lists.append(bit_list)

            # schedule another
            self.start_time = self.own.timeline.now()
            process = Process(self, "begin_photon_pulse", [])
            event = Event(self.start_time + int(round(self.light_time * 1e12)), process)
            self.own.timeline.schedule(event)

        else:
            self.working = False
            self.another.working = False

            self.key_lengths.pop(0)
            self.keys_left_list.pop(0)
            self.end_run_times.pop(0)
            self.another.key_lengths.pop(0)
            self.another.end_run_times.pop(0)

            # wait for quantum channel to clear of photons, then start protocol
            time = self.own.timeline.now() + self.own.qchannels[self.another.own.name].delay + 1
            process = Process(self, "start_protocol", [])
            event = Event(time, process)
            self.own.timeline.schedule(event)

    def set_measure_basis_list(self) -> None:
        """Method to set measurement basis list."""

        log.logger.debug(self.name + " setting measurement basis")

        num_pulses = int(self.light_time * self.ls_freq)
        basis_list = numpy.random.choice([0, 1], num_pulses)
        self.basis_lists.append(basis_list)
        self.own.components[self.qsd_name].set_basis_list(basis_list, self.start_time, self.ls_freq)

    def end_photon_pulse(self) -> None:
        """Method to process sent qubits."""

        log.logger.debug(self.name + " ending photon pulse")

        if self.working and self.own.timeline.now() < self.end_run_times[0]:
            # get bits
            self.bit_lists.append(self.own.get_bits(self.light_time, self.start_time, self.ls_freq, self.qsd_name))
            self.start_time = self.own.timeline.now()
            # set bases for measurement
            self.set_measure_basis_list()

            # schedule another if necessary
            if self.own.timeline.now() + self.light_time * 1e12 - 1 < self.end_run_times[0]:
                # schedule another
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + int(round(self.light_time * 1e12) - 1), process)
                self.own.timeline.schedule(event)

            # send message that we got photons
            message = SARG04Message(SARG04MsgType.RECEIVED_QUBITS, self.another.name)
            self.own.send_message(self.another.own.name, message)

    def received_message(self, src: str, msg: "Message") -> None:
        """Method to receive messages.

        Will perform different processing actions based on the message received.

        Args:
            src (str): source node sending message.
            msg (Message): message received.
        """

        if self.working and self.own.timeline.now() < self.end_run_times[0]:
            if msg.msg_type is SARG04MsgType.BEGIN_PHOTON_PULSE:  # (current node is Bob): start to receive photons
                self.ls_freq = msg.frequency
                self.light_time = msg.light_time

                log.logger.debug(self.name + " received BEGIN_PHOTON_PULSE, ls_freq={}, light_time={}".format(self.ls_freq, self.light_time))

                self.start_time = int(msg.start_time) + self.own.qchannels[src].delay

                # generate and set basis list
                self.set_measure_basis_list()

                # schedule end_photon_pulse()
                process = Process(self, "end_photon_pulse", [])
                event = Event(self.start_time + round(self.light_time * 1e12) - 1, process)
                self.own.timeline.schedule(event)

            elif msg.msg_type is SARG04MsgType.RECEIVED_QUBITS:  # (Current node is Alice): can secret bit
                log.logger.debug(self.name + " received RECEIVED_QUBITS message")
                bases = self.basis_lists.pop(0) # pulling stored list of bases
                states = self.bit_lists[0] # generating states based off of alice's bits
                conjugate = list(zip(states, bases)) # creating a combined list of state + basis
                
                # Generating the "false" state
                # random state + opposite of basis list
                # form: [(1, 0), (0, 1), (0, 0) ...]
                non_orthogonal = list(zip(numpy.random.choice([0, 1], len(bases)), [b^1 for b in bases]))

                message = SARG04Message(SARG04MsgType.BASIS_LIST, self.another.name, bases=conjugate, non_orthogonal=non_orthogonal)
                self.own.send_message(self.another.own.name, message)

            elif msg.msg_type is SARG04MsgType.BASIS_LIST:  # (Current node is Bob): compare bases
                log.logger.debug(self.name + " received BASIS_LIST message")
                
                # parse alice basis list
                alice_basis = msg.bases 
                other_basis = msg.non_orthogonal

                # compare own basis with basis message and create list of matching indices
                indices = []
                basis_list = self.basis_lists.pop(0)
                bits = self.bit_lists.pop(0)

                # mapping bob's observed state and basis to an index for the lookup table
                observe_indexes = {
                    (0,1): 0,
                    (1,1): 1,
                    (0,0): 2,
                    (1,0): 3
                }
                # mapping the potential combinations of states that Alice can announce
                announce_indexes = [[(0,0), (0,1)],[(0,1), (1,0)], [(0,0), (1,1)],[(1,0), (1,1)]]
                
                for i,observed in enumerate(zip(bits,basis_list)):
                    bit, basis = observed # Whatever Bob observes
                    # bit == -1 indicates some kind of error
                    if bit != -1:
                        key1 = observe_indexes[(bit, basis)]
                        key2 = announce_indexes.index(sorted([alice_basis[i], other_basis[i]]))
                        # If it's possible to draw a conclusion, we save that index
                        if self.lookup_table[key1][key2] is not None:
                            indices.append(i)
                            self.key_bits.append(self.lookup_table[key1][key2][0])
                
                # send to Alice list of matching indices
                message = SARG04Message(SARG04MsgType.MATCHING_INDICES, self.another.name, indices=indices)
                self.own.send_message(self.another.own.name, message)

            elif msg.msg_type is SARG04MsgType.MATCHING_INDICES:  # (Current node is Alice): create key from matching indices
                log.logger.debug(self.name + " received MATCHING_INDICES message")
                # parse matching indices
                indices = msg.indices

                bits = self.bit_lists.pop(0)

                # set key equal to bits at received indices
                for i in indices:
                    self.key_bits.append(bits[i])

                # check if key long enough. If it is, truncate if necessary and call cascade
                if len(self.key_bits) >= self.key_lengths[0]:
                    throughput = self.key_lengths[0] * 1e12 / (self.own.timeline.now() - self.last_key_time)

                    while len(self.key_bits) >= self.key_lengths[0] and self.keys_left_list[0] > 0:
                        log.logger.info(self.name + " generated a valid key")
                        self.set_key()  # convert from binary list to int
                        self._pop(info=self.key)
                        self.another.set_key()
                        self.another._pop(info=self.another.key)  # TODO: why access another node?

                        # for metrics
                        if self.latency == 0:
                            self.latency = (self.own.timeline.now() - self.last_key_time) * 1e-12

                        self.throughputs.append(throughput)

                        key_diff = self.key ^ self.another.key
                        num_errors = 0
                        while key_diff:
                            key_diff &= key_diff - 1
                            num_errors += 1
                        self.error_rates.append(num_errors / self.key_lengths[0])

                        self.keys_left_list[0] -= 1

                    self.last_key_time = self.own.timeline.now()

                # check if we're done
                if self.keys_left_list[0] < 1:
                    self.working = False
                    self.another.working = False


    def set_key(self):
        """Method to convert `bit_list` field (List[int]) to a single key (int)."""

        key_bits = self.key_bits[0:self.key_lengths[0]]
        del self.key_bits[0:self.key_lengths[0]]
        self.key = int("".join(str(x) for x in key_bits), 2)  # convert from binary list to int

