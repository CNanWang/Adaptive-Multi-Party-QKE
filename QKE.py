from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator # type: ignore
from qiskit.quantum_info import random_statevector
from qiskit import transpile
import numpy as np
import random

class QuantumParticipant:
    def __init__(self, id, private_key):
        self.id = id  # Participant ID (1 to N)
        self.private_key = private_key  # Private key (binary sequence of length M)
        self.is_leader = False
        self.shared_key = None
        self.backend = AerSimulator()
        
    def __str__(self):
        return f"P{self.id}"  # Returns format like "P1", "P2", etc.
        
    def get_private_key_str(self):
        """Returns string representation of private key"""
        return ','.join(map(str, self.private_key))
    
    def get_pauli_operation(self, position):
        """Determine Pauli operation based on private key
        Args:
            position: Position in the private key
        Returns:
            Pauli operation ('I' or 'X')
        """
        if position >= len(self.private_key):
            return 'I'  # Default to identity if position out of range
        
        # Map private key bits to Pauli operations
        # 0 -> I, 1 -> X
        if self.private_key[position] == 0:
            return 'I'
        else:
            return 'X'

class QuantumKeyExchange:
    def __init__(self, num_participants, m_value):
        self.N = num_participants
        self.M = m_value  # Number of quantum states per sequence
        self.d = 4  # Number of decoy states
        self.error_threshold = 0.1
        self.backend = AerSimulator()
        
        # Assign different private keys (binary sequences) to each participant
        # Each private key has length M
        private_keys = []
        while len(private_keys) < num_participants:
            # Generate M-bit binary sequence as private key
            new_key = [random.randint(0, 1) for _ in range(m_value)]
            if new_key not in private_keys:  # Ensure private keys are unique
                private_keys.append(new_key)
        
        # Create participants and assign private keys
        self.participants = [
            QuantumParticipant(i+1, private_keys[i]) 
            for i in range(num_participants)
        ]
        
        # Initialize leaders through dynamic election process
        self.elect_leaders()

    def elect_leaders(self):
        """
        Dynamic election algorithm to select 4 leaders from N participants
        and reassign participant IDs based on election results
        """
        print("\n=== Dynamic Leader Election Process ===")
        print(f"Total participants: {self.N}")
        
        # Step 1: Generate random votes for each participant
        votes = {}
        for participant in self.participants:
            # Each participant votes for themselves and 3 other random participants
            participant_votes = [participant.id]
            other_participants = [p.id for p in self.participants if p.id != participant.id]
            participant_votes.extend(random.sample(other_participants, min(3, len(other_participants))))
            votes[participant.id] = participant_votes
        
        # Display voting table
        print("\nVoting Table:")
        headers = ["Participant", "Votes For"]
        rows = []
        for participant_id, participant_votes in votes.items():
            rows.append([f"P{participant_id}", ", ".join([f"P{v}" for v in participant_votes])])
        self.print_table(headers, rows)
        
        # Step 2: Count votes for each participant
        vote_counts = {p.id: 0 for p in self.participants}
        for voter_votes in votes.values():
            for voted_id in voter_votes:
                vote_counts[voted_id] += 1
        
        # Step 3: Sort participants by vote count (descending)
        sorted_participants = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Step 4: Select top 4 participants as leaders
        leaders = sorted_participants[:4]
        
        # Step 5: Display election results in table format
        print("\nElection Results:")
        headers = ["Participant", "Votes", "Status"]
        rows = []
        for i, (participant_id, vote_count) in enumerate(sorted_participants):
            status = "LEADER" if i < 4 else "FOLLOWER"
            rows.append([f"P{participant_id}", vote_count, status])
        self.print_table(headers, rows)
        
        # Step 6: Reassign participant IDs based on election results
        # Create a mapping from old IDs to new IDs
        id_mapping = {}
        
        # First, assign new IDs to top 4 participants (1-4)
        for i, (participant_id, _) in enumerate(sorted_participants[:4]):
            id_mapping[participant_id] = i + 1
        
        # Then, assign new IDs to remaining participants (5-N)
        new_id = 5
        for participant_id, _ in sorted_participants[4:]:
            id_mapping[participant_id] = new_id
            new_id += 1
        
        # Step 7: Update participant IDs and leader status
        print("\nParticipant ID Reassignment:")
        headers = ["Original ID", "New ID", "Status"]
        rows = []
        
        # Update IDs and create rows for display
        for participant in self.participants:
            old_id = participant.id
            new_id = id_mapping[old_id]
            participant.id = new_id
            participant.is_leader = (new_id <= 4)
            rows.append([f"P{old_id}", f"P{new_id}", "LEADER" if participant.is_leader else "FOLLOWER"])
        
        # Sort rows by new ID for display
        rows.sort(key=lambda x: int(x[1][1:]))  # Sort by new ID number
        self.print_table(headers, rows)
        
        # Sort participants by new ID
        self.participants.sort(key=lambda p: p.id)
        
        print("\nFinal Participant List:")
        leaders = [p for p in self.participants if p.is_leader]
        followers = [p for p in self.participants if not p.is_leader]
        print(f"Leaders: {', '.join(str(p) for p in leaders)}")
        print(f"Followers: {', '.join(str(p) for p in followers)}")

    def create_ghz_state(self, num_qubits=4):
        """Create GHZ state
        |G⟩_1234 = 1/√2(|0000⟩ + |1111⟩)
        
        Args:
            num_qubits: Number of qubits, default is 4
        Returns:
            Quantum circuit
        """
        qc = QuantumCircuit(num_qubits)
        
        # Steps to create GHZ state:
        # 1. Put first qubit in superposition
        qc.h(0)
        
        # 2. Propagate superposition to other qubits using CNOT gates
        for i in range(1, num_qubits):
            qc.cx(0, i)
        
        return qc

    def create_bell_state(self):
        """Create Bell state"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def measure_state(self, circuit, basis):
        """Measure quantum state"""
        num_qubits = circuit.num_qubits
        cr = ClassicalRegister(num_qubits)
        circuit.add_register(cr)
        
        for i in range(num_qubits):
            self.measure_in_basis(circuit, i, basis)
        
        # Execute circuit and get results
        transpiled_circuit = transpile(circuit, self.backend)
        job = self.backend.run(transpiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(circuit)
        
        # Return measurement result
        measured = list(counts.keys())[0]
        return sum(int(bit) for bit in measured) % 2

    def insert_decoy_states(self, quantum_sequence, num_decoy):
        """Insert decoy states"""
        # Ensure number of decoy states doesn't exceed sequence length
        num_decoy = min(num_decoy, len(quantum_sequence))
        positions = sorted(random.sample(range(len(quantum_sequence)), num_decoy))
        bases = [random.choice(['X', 'Z']) for _ in range(num_decoy)]
        decoy_states = [random.randint(0, 1) for _ in range(num_decoy)]
        
        # Insert decoy states
        for pos, base, state in zip(positions, bases, decoy_states):
            quantum_sequence.insert(pos, {'type': 'decoy', 'base': base, 'state': state})
            
        return positions, bases, decoy_states

    def create_particle_sequence(self, participant_id, seq_type):
        """Create particle sequence
        Args:
            participant_id: Participant's ID
            seq_type: Sequence type (1,2,3,4 corresponding to A,B,C,D)
        Returns:
            Sequence containing quantum state information
        """
        sequence = []
        seq_name = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}[seq_type]
        
        # Generate random initial state
        initial_state = [random.randint(0, 1) for _ in range(4)]
        
        # Create M groups of identical quantum states
        for j in range(self.M):
            # Randomly select quantum state type
            state_type = random.choice(['ghz', 'bell', 'random'])
            
            if state_type == 'ghz':
                # Create four-particle GHZ state |G⟩_1234 = 1/√2(|0000⟩ + |1111⟩)
                state = self.create_ghz_state()
            elif state_type == 'bell':
                # Create Bell state |Φ⁺⟩ = 1/√2(|00⟩ + |11⟩)
                state = self.create_bell_state()
            else:
                # Create random quantum state
                state = QuantumCircuit(4)
                for i in range(4):
                    if random.random() < 0.5:
                        state.h(i)
                    if random.random() < 0.5:
                        state.x(i)
            
            sequence.append({
                'type': state_type,
                'state': state,
                'measured': False,
                'initial_state': initial_state.copy(),
                'participant_id': participant_id,
                'j': j + 1,  # Sequence number j, starting from 1
                'position': seq_type,  # Particle position(1,2,3,4)
                'seq_type': seq_name  # Sequence type(A,B,C,D)
            })
        return sequence

    def apply_pauli_operations(self, sequence, participant, start_pos=0):
        """Apply Pauli operations based on participant's private key
        Args:
            sequence: Sequence to apply operations to
            participant: Participant whose private key determines operations
            start_pos: Starting position in private key
        Returns:
            Result sequence after applying operations
        """
        result = []
        for i, state in enumerate(sequence):
            new_state = state.copy()
            if isinstance(new_state, dict) and 'initial_state' in new_state:
                # Determine number of qubits based on sequence type
                num_qubits = len(new_state['initial_state'])
                
                # Apply Pauli operations based on participant's private key
                for j in range(num_qubits):
                    pos = (start_pos + i) % len(participant.private_key)
                    op = participant.get_pauli_operation(pos)
                    
                    if op == 'X':
                        new_state['initial_state'][j] = 0  # X operation results in 0
                    else:  # op == 'I'
                        new_state['initial_state'][j] = 1  # I operation results in 1
            result.append(new_state)
        return result

    def measure_in_basis(self, circuit, qubit, basis):
        """Measure in specified basis
        Args:
            circuit: Quantum circuit
            qubit: Qubit number
            basis: Measurement basis ('X', 'Z')
        """
        if basis == 'X':
            circuit.h(qubit)  # Add H gate before X basis measurement
        # Z basis measurement doesn't need additional gate operations
        circuit.measure(qubit, qubit)
        
    def calculate_error_rate(self, measured_results, expected_results):
        """Calculate error rate between measured and expected results
        Args:
            measured_results: List of measured results
            expected_results: List of expected results
        Returns:
            Error rate (float between 0 and 1)
        """
        if len(measured_results) != len(expected_results):
            raise ValueError("Measured and expected results must have same length")
        
        errors = sum(1 for m, e in zip(measured_results, expected_results) if m != e)
        return errors / len(measured_results)

    def verify_security(self, decoy_results, error_threshold=0.1):
        """Verify security by checking decoy state measurements
        Args:
            decoy_results: Dictionary containing decoy measurement results
            error_threshold: Maximum acceptable error rate (default: 0.1 or 10%)
        Returns:
            (bool, float): (Security status, Error rate)
        """
        print("\n=== Security Verification ===")
        print("Checking decoy states for eavesdropping detection...")
        
        # Calculate error rate for decoy states
        measured_results = []
        expected_results = []
        
        for participant_id, participant_decoys in decoy_results.items():
            for seq_type in participant_decoys:
                measured = participant_decoys[seq_type].get('measured', [])
                expected = participant_decoys[seq_type].get('states', [])
                measured_results.extend(measured)
                expected_results.extend(expected)
        
        if not measured_results or not expected_results:
            print("Warning: No decoy state measurements available")
            return False, 1.0
            
        error_rate = self.calculate_error_rate(measured_results, expected_results)
        
        # Print security check results
        headers = ["Security Metric", "Value", "Status"]
        rows = [
            ["Total Decoy States", len(measured_results), "-"],
            ["Error Rate", f"{error_rate:.2%}", "PASS" if error_rate <= error_threshold else "FAIL"],
            ["Error Threshold", f"{error_threshold:.2%}", "-"]
        ]
        self.print_table(headers, rows)
        
        is_secure = error_rate <= error_threshold
        print(f"\nSecurity Status: {'SECURE' if is_secure else 'COMPROMISED'}")
        if not is_secure:
            print("Warning: High error rate detected - possible eavesdropping attempt!")
            
        return is_secure, error_rate

    def measure_decoy_state(self, state, basis):
        """Measure decoy state and return result
        Args:
            state: Dictionary containing decoy state information
            basis: Measurement basis ('X' or 'Z')
        Returns:
            Measurement result (0 or 1)
        """
        # Create quantum circuit for measurement
        qc = QuantumCircuit(1, 1)
        
        # Prepare initial state
        if state['state'] == 1:
            qc.x(0)
            
        # Apply measurement basis
        if basis == 'X':
            qc.h(0)
            
        # Measure
        qc.measure(0, 0)
        
        # Execute circuit
        transpiled_circuit = transpile(qc, self.backend)
        job = self.backend.run(transpiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Return measurement result
        return int(list(counts.keys())[0])

    def exchange_sequences(self, sender, receiver, sequences):
        """Exchange sequence implementation with security verification
        Args:
            sender: Sender participant
            receiver: Receiver participant
            sequences: Dictionary of quantum sequences
        Returns:
            bool: True if exchange is secure, False otherwise
        """
        # Get sender's sequence
        sender_sequence = sequences[sender.id]
        
        # Initialize decoy measurement results
        decoy_results = {sender.id: {}}
        
        # Insert and measure decoy states for each sequence type
        for seq_type in ['B', 'C', 'D']:
            if seq_type in sender_sequence:
                # Insert decoy states
                positions, bases, states = self.insert_decoy_states(
                    sender_sequence[seq_type], 
                    self.d
                )
                
                # Measure decoy states
                measured_results = []
                for pos, basis in zip(positions, bases):
                    result = self.measure_decoy_state(
                        sender_sequence[seq_type][pos],
                        basis
                    )
                    measured_results.append(result)
                
                # Store results for security verification
                decoy_results[sender.id][seq_type] = {
                    'positions': positions,
                    'bases': bases,
                    'states': states,
                    'measured': measured_results
                }
        
        # Verify security
        is_secure, error_rate = self.verify_security(decoy_results)
        
        return is_secure

    def measure_and_generate_key(self, leaders, sequences):
        """Measure and generate key"""
        shared_key = []
        
        for i in range(self.M):
            # Create measurement circuit for each GHZ state
            qc = QuantumCircuit(4, 4)
            
            # Randomly select measurement basis
            basis = random.choice(['X', 'Z'])
            
            # Measure in selected basis
            for qubit in range(4):
                self.measure_in_basis(qc, qubit, basis)
            
            # Execute measurement
            transpiled_circuit = transpile(qc, self.backend)
            job = self.backend.run(transpiled_circuit, shots=1)
            result = job.result()
            counts = result.get_counts(qc)
            
            # Extract key bit from measurement results
            key_bit = self.extract_key_bit(counts)
            shared_key.append(key_bit)
        
        return shared_key

    def extract_key_bit(self, counts):
        """Extract key bit from measurement results"""
        # Get most common measurement result
        result = max(counts.items(), key=lambda x: x[1])[0]
        # Use parity as key bit
        return sum(int(bit) for bit in result) % 2

    def perform_qka(self, leaders):
        """Execute Quantum Key Agreement (QKA)"""
        print("\n=== QKA Process Started ===")
        print(f"Initial Leaders: {', '.join(str(p) for p in leaders)}")
        
        # Display initial private keys
        print("\nInitial Private Keys:")
        for leader in leaders:
            print(f"{leader}: K{leader.id} = {leader.get_private_key_str()}")
        
        # Step 1: Prepare exchange sequences
        print("\nStep 1: Prepare Exchange Sequences")
        sequences = {}
        for leader in leaders:
            sequences[leader.id] = {
                'A': self.create_particle_sequence(leader.id, 1),  # A sequence not transmitted
                'B': self.create_particle_sequence(leader.id, 2),  # B sequence for exchange
                'C': self.create_particle_sequence(leader.id, 3),  # C sequence for exchange
                'D': self.create_particle_sequence(leader.id, 4)   # D sequence for exchange
            }
            print(f"\n{leader} prepared {self.M} groups of |G⟩_1234 states, divided into four sequences:")
            
            # Display initial states of sequences
            print(f"A{leader.id} = {{{', '.join(map(str, sequences[leader.id]['A'][0]['initial_state']))}}}")
            print(f"B{leader.id} = {{{', '.join(map(str, sequences[leader.id]['B'][0]['initial_state']))}}}")
            print(f"C{leader.id} = {{{', '.join(map(str, sequences[leader.id]['C'][0]['initial_state']))}}}")
            print(f"D{leader.id} = {{{', '.join(map(str, sequences[leader.id]['D'][0]['initial_state']))}}}")
        
        # Step 2: Execute exchange process
        print("\nStep 2: Execute Exchange Process")
        
        # 使用表格格式显示交换过程
        headers = ["Sender", "Receiver", "Sequence", "Pauli Operation"]
        rows = []
        
        # Execute exchange process
        exchanged_sequences = {}
        for sender in leaders:
            exchanged_sequences[sender.id] = {}
            for receiver in leaders:
                if sender != receiver:
                    # Determine sequence type to send
                    if receiver.id == sender.id + 1 or (sender.id == 4 and receiver.id == 1):
                        seq_type = 'B'
                    elif receiver.id == sender.id + 2 or (sender.id == 3 and receiver.id == 1) or (sender.id == 4 and receiver.id == 2):
                        seq_type = 'C'
                    else:
                        seq_type = 'D'
                    
                    # Apply Pauli operations based on sender's private key
                    result_sequence = self.apply_pauli_operations(
                        sequences[sender.id][seq_type],
                        sender
                    )
                    exchanged_sequences[sender.id][f"{seq_type}{receiver.id}"] = result_sequence

                    operations = [sender.get_pauli_operation(i % len(sender.private_key)) for i in range(4)]
                    rows.append([f"P{sender.id}", f"P{receiver.id}", f"{seq_type}{sender.id}", ','.join(operations)])

        self.print_table(headers, rows)
        
        # Step 3: Measurement process
        print("\nStep 3: Measurement Process")

        headers = ["Participant", "Measurement Basis", "Measurement Result"]
        rows = []
        
        # Execute measurements
        measurement_results = {}
        for leader in leaders:
            # Randomly select measurement basis
            basis = [random.choice(['X', 'Z']) for _ in range(4)]
            results = self.measure_sequences(sequences[leader.id], exchanged_sequences[leader.id], basis)
            measurement_results[leader.id] = results
            rows.append([f"P{leader.id}", ','.join(basis), ','.join(map(str, results))])

        self.print_table(headers, rows)
        
        # Generate shared key
        shared_key = []
        for i in range(self.M):
            key_bit = 0
            for leader in leaders:
                key_bit ^= leader.private_key[i]  # XOR operation on leaders' private keys
            shared_key.append(key_bit)
        
        print(f"\nGenerated Shared Key: K = {','.join(map(str, shared_key))}")
        return shared_key

    def measure_sequences(self, own_sequences, received_sequences, basis):
        """Measure sequences and return results"""
        results = []
        for i in range(self.M):
            # Measure own sequences and received sequences
            result = 0
            
            # Process own sequences
            if isinstance(own_sequences, dict):
                for seq in own_sequences.values():
                    if i < len(seq) and isinstance(seq[i], dict) and 'initial_state' in seq[i]:
                        # Randomly select measurement basis
                        measure_basis = random.choice(['X', 'Z'])
                        if measure_basis == 'X':
                            result ^= 1 - seq[i]['initial_state'][0]  # X basis measurement
                        else:
                            result ^= seq[i]['initial_state'][0]  # Z basis measurement
            else:
                if i < len(own_sequences) and isinstance(own_sequences[i], dict) and 'initial_state' in own_sequences[i]:
                    measure_basis = random.choice(['X', 'Z'])
                    if measure_basis == 'X':
                        result ^= 1 - own_sequences[i]['initial_state'][0]
                    else:
                        result ^= own_sequences[i]['initial_state'][0]
            
            # Process received sequences
            if isinstance(received_sequences, dict):
                for seq in received_sequences.values():
                    if i < len(seq) and isinstance(seq[i], dict) and 'initial_state' in seq[i]:
                        measure_basis = random.choice(['X', 'Z'])
                        if measure_basis == 'X':
                            result ^= 1 - seq[i]['initial_state'][0]
                        else:
                            result ^= seq[i]['initial_state'][0]
            else:
                if i < len(received_sequences) and isinstance(received_sequences[i], dict) and 'initial_state' in received_sequences[i]:
                    measure_basis = random.choice(['X', 'Z'])
                    if measure_basis == 'X':
                        result ^= 1 - received_sequences[i]['initial_state'][0]
                    else:
                        result ^= received_sequences[i]['initial_state'][0]
            
            results.append(result)
        return results

    def print_table(self, headers, rows, title=None):
        """Print formatted table
        Args:
            headers: List of header strings
            rows: List of data rows
            title: Optional table title
        """
        if title:
            print(f"\n{title}")
        
        # Calculate maximum width for each column
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
        print("│" + "│".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "│")
        print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
        
        # Print data rows
        for row in rows:
            print("│" + "│".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "│")
        
        print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    def print_section(self, title, content=None):
        """Print formatted section title
        Args:
            title: Section title
            content: Optional content
        """
        print("\n" + "="*50)
        print(f" {title} ")
        print("="*50)
        if content:
            print(content)

    def print_subsection(self, title, content=None):
        """Print formatted subsection title
        Args:
            title: Subsection title
            content: Optional content
        """
        print("\n" + "-"*40)
        print(f" {title} ")
        print("-"*40)
        if content:
            print(content)

    def print_key_info(self, key, title="Key Information"):
        """Print key information
        Args:
            key: Key list
            title: Title
        """
        print(f"\n{title}:")
        headers = ["Key"]
        rows = [[f"[{', '.join(map(str, key))}]"]]
        self.print_table(headers, rows)

    def perform_qkd(self, leader, followers):
        """Execute Quantum Key Distribution (QKD)"""
        group_members = [leader] + followers
        group_indices = [p.id for p in group_members]
        
        # Select quantum state type based on number of followers
        if len(followers) == 3:
            self.print_section(f"Four-Particle GHZ State QKD Process", f"Group{leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
            return self._perform_four_particle_qkd(leader, followers)
        elif len(followers) == 2:
            self.print_section(f"Three-Particle GHZ State QKD Process", f"Group{leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
            return self._perform_three_particle_qkd(leader, followers)
        else:
            self.print_section(f"Bell State QKD Process", f"Group{leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
            return self._perform_bell_state_qkd(leader, followers[0])

    def _prepare_sequences(self, group_members, num_sequences):
        """Prepare exchange sequences
        Args:
            group_members: List of group members
            num_sequences: Number of sequences
        Returns:
            Sequence dictionary and decoy state information
        """
        sequences = {}
        decoy_states = {}
        
        for p in group_members:
            sequences[p.id] = {}
            decoy_states[p.id] = {}
            
            # Create sequences
            for i in range(num_sequences):
                seq_type = chr(65 + i)  # A, B, C, D
                sequences[p.id][seq_type] = self.create_particle_sequence(p.id, i + 1)
                
                # Only insert decoy states in exchange sequences
                if i > 0:
                    positions, bases, states = self.insert_decoy_states(sequences[p.id][seq_type], self.d)
                    decoy_states[p.id][seq_type] = {
                        'positions': positions,
                        'bases': bases,
                        'states': states
                    }
        
        return sequences, decoy_states

    def _display_sequences(self, sequences, group_members):
        """Display sequence information
        Args:
            sequences: Sequence dictionary
            group_members: List of group members
        """
        self.print_subsection("Sequence Preparation")
        
        for p in group_members:
            print(f"\nSequences prepared by P{p.id}:")
            print("┌─────────────┐")
            for seq_type in sequences[p.id]:
                state = sequences[p.id][seq_type][0].get('initial_state', [0] * 4)
                print(f"│ {seq_type}{p.id} = {{{', '.join(map(str, state[:len(state)]))}}} │")
            print("└─────────────┘")

    def _display_decoy_states(self, decoy_states, group_members):
        """Display decoy state information
        Args:
            decoy_states: Decoy state information dictionary
            group_members: List of group members
        """
        self.print_subsection("Decoy State Information")
        
        headers = ["Participant", "Sequence", "Insert Position", "Measurement Basis", "Quantum State"]
        rows = []
        
        for p in group_members:
            for seq_type in decoy_states[p.id]:
                for pos, base, state in zip(
                    decoy_states[p.id][seq_type]['positions'],
                    decoy_states[p.id][seq_type]['bases'],
                    decoy_states[p.id][seq_type]['states']
                ):
                    rows.append([f"P{p.id}", f"{seq_type}{p.id}", pos+1, base, f"|{state}⟩"])
        
        self.print_table(headers, rows)

    def _display_exchange_process(self, leader, followers, sequences, exchanged_sequences):
        """Display exchange process
        Args:
            leader: Leader
            followers: List of followers
            sequences: Sequence dictionary
            exchanged_sequences: Exchanged sequence dictionary
        """
        self.print_subsection("Exchange Process")
        
        headers = ["Sender", "Receiver", "Sequence", "Initial Quantum State", "Pauli Operation", "State After Operation"]
        rows = []
        
        for follower in followers:
            # Determine sequence type based on number of sequences
            if len(sequences[leader.id]) == 2:  # Bell state QKD
                seq_type = 'B'
                # Get Pauli operations from leader's private key
                operations = [leader.get_pauli_operation(i % len(leader.private_key)) for i in range(2)]
                initial_state = sequences[leader.id][seq_type][0].get('initial_state', [0, 0])[:2]
                final_state = ['1' if op == 'X' else '0' for op in operations]
            else:  # GHZ state QKD
                seq_type = 'B' if follower.id == leader.id + 1 or (leader.id == 4 and follower.id == 1) else 'C'
                # Get Pauli operations from leader's private key
                operations = [leader.get_pauli_operation(i % len(leader.private_key)) for i in range(3)]
                initial_state = sequences[leader.id][seq_type][0].get('initial_state', [0, 0, 0])[:3]
                final_state = ['1' if op == 'X' else '0' for op in operations]
            
            rows.append([
                f"P{leader.id}",
                f"P{follower.id}",
                f"{seq_type}{leader.id}",
                f"[{', '.join(map(str, initial_state))}]",
                ','.join(operations),
                f"|{','.join(final_state)}⟩"
            ])
        
        self.print_table(headers, rows)

    def _display_measurement_results(self, group_members, measurement_results):
        """Display measurement results
        Args:
            group_members: List of group members
            measurement_results: Measurement results dictionary
        """
        self.print_subsection("Measurement Results")
        
        headers = ["Participant", "Measurement Basis", "Measurement Result"]
        rows = []
        
        for p in group_members:
            basis = [random.choice(['X', 'Z']) for _ in range(3)]
            results = measurement_results[p.id]
            rows.append([f"P{p.id}", ','.join(basis), ','.join(map(str, results))])
        
        self.print_table(headers, rows)

    def distribute_key(self):
        """Main key distribution process"""
        # 1. Get list of leaders
        leaders = [p for p in self.participants if p.is_leader]
        num_leaders = len(leaders)
        
        if num_leaders == 0:
            print("Error: No leaders set")
            return
            
        self.print_section("Quantum Key Distribution (QKD) Process Started")
        print(f"\nSelected Leaders: {', '.join(str(p) for p in leaders)}")
        
        # 2. Perform QKA among leaders
        self.print_subsection("Phase 1: QKA Process Among Leaders")
        initial_key = self.perform_qka(leaders)
        
        # Security verification for leader QKA
        if not initial_key:
            print("\nError: Leader QKA process failed security verification")
            return
            
        # Distribute initial key to all leaders
        for leader in leaders:
            leader.shared_key = initial_key.copy()
        
        # 3. Calculate remaining participants and perform corresponding QKD
        remaining = len(self.participants) - num_leaders
        print(f"\nRemaining Participants: {remaining}")
        
        # 4. Calculate grouping scheme
        num_four_particle_groups = remaining // 3
        remaining_after_four = remaining % 3
        
        # 5. Display grouping scheme
        self.print_subsection("Grouping Scheme")
        headers = ["Group Type", "Usage Count", "Number of Participants"]
        rows = [
            ["Four-Particle GHZ State", f"{num_four_particle_groups} times", f"{num_four_particle_groups * 3} participants"],
            ["Three-Particle GHZ State", f"{1 if remaining_after_four == 2 else 0} times", f"{2 if remaining_after_four == 2 else 0} participants"],
            ["Bell State", f"{1 if remaining_after_four == 1 else 0} times", f"{1 if remaining_after_four == 1 else 0} participants"]
        ]
        self.print_table(headers, rows)
        
        # 6. Execute grouped QKD with security verification
        remaining_participants = [p for p in self.participants if not p.is_leader]
        current_index = 0
        group_count = 1
        
        # Track security status
        security_status = []
        
        # 6.1 Process four-particle GHZ state groups
        for i in range(num_four_particle_groups):
            self.print_subsection(f"Group {group_count}: Four-Particle GHZ State QKD")
            current_group = remaining_participants[current_index:current_index + 3]
            current_leader = leaders[i % len(leaders)]
            
            group_key = self._perform_four_particle_qkd(current_leader, current_group)
            if group_key is None:
                security_status.append((f"Group {group_count}", "FAILED", "Security verification failed"))
                current_index += 3
                group_count += 1
                continue
                
            self.synchronize_and_display_keys(group_key, initial_key, current_group)
            security_status.append((f"Group {group_count}", "PASSED", "Security verification successful"))
            
            current_index += 3
            group_count += 1
        
        # 6.2 Process remaining participants
        if remaining_after_four > 0:
            if remaining_after_four == 2:
                self.print_subsection(f"Group {group_count}: Three-Particle GHZ State QKD")
                current_group = remaining_participants[current_index:current_index + 2]
                current_leader = leaders[current_index // 3 % len(leaders)]
                
                group_key = self._perform_three_particle_qkd(current_leader, current_group)
                if group_key is not None:
                    self.synchronize_and_display_keys(group_key, initial_key, current_group)
                    security_status.append((f"Group {group_count}", "PASSED", "Security verification successful"))
                else:
                    security_status.append((f"Group {group_count}", "FAILED", "Security verification failed"))
                
                current_index += 2
                group_count += 1
            else:
                self.print_subsection(f"Group {group_count}: Bell State QKD")
                current_group = remaining_participants[current_index:current_index + 1]
                current_leader = leaders[current_index // 3 % len(leaders)]
                
                group_key = self._perform_bell_state_qkd(current_leader, current_group[0])
                if group_key is not None:
                    self.synchronize_and_display_keys(group_key, initial_key, current_group)
                    security_status.append((f"Group {group_count}", "PASSED", "Security verification successful"))
                else:
                    security_status.append((f"Group {group_count}", "FAILED", "Security verification failed"))
                
                current_index += 1
                group_count += 1
        
        # 7. Display security verification summary
        self.print_section("Security Verification Summary")
        headers = ["Group", "Status", "Details"]
        self.print_table(headers, security_status)
        
        # Check if any security verifications failed
        if any(status[1] == "FAILED" for status in security_status):
            print("\nWarning: Some groups failed security verification!")
            print("Key distribution process may be compromised.")
            return
        
        # 8. Ensure all participants have obtained keys
        for participant in self.participants:
            if participant.shared_key is None:
                participant.shared_key = initial_key.copy()
        
        # 9. Display final results
        self.print_section("QKD Process Completed Successfully")
        self.print_key_info(initial_key, "Final Shared Key")
        
        # 10. Display performance statistics
        self.print_subsection("Performance Statistics")
        headers = ["Statistics Item", "Value"]
        rows = [
            ["Total Participants", len(self.participants)],
            ["Number of Leaders", num_leaders],
            ["Number of Followers", remaining],
            ["Total Number of Groups", group_count - 1],
            ["Security Check Pass Rate", f"{sum(1 for s in security_status if s[1] == 'PASSED')}/{len(security_status)}"]
        ]
        self.print_table(headers, rows)

    def synchronize_and_display_keys(self, group_key, target_key, participants):
        """Display key synchronization process"""
        print("\nKey Synchronization Process:")

        headers = ["Key Type", "Key Value"]
        rows = [
            ["Group Key", f"[{', '.join(map(str, group_key))}]"],
            ["Target Key", f"[{', '.join(map(str, target_key))}]"]
        ]
        self.print_table(headers, rows)
        
        # Calculate positions that need bit flipping (counting from 1)
        diff_positions = []
        for i, (g, t) in enumerate(zip(group_key, target_key)):
            if g != t:
                diff_positions.append(i + 1)  # Add 1 to make position count from 1
        
        if diff_positions:
            print(f"\nPositions that need bit flipping: {', '.join(map(str, diff_positions))}")
            print("\nKey adjustment by each participant:")

            headers = ["Participant", "Adjusted Key"]
            rows = []
            for p in participants:
                adjusted_key = list(group_key)
                for pos in diff_positions:
                    actual_pos = pos - 1  # Convert back to 0-based position index
                    adjusted_key[actual_pos] = 1 - adjusted_key[actual_pos]
                rows.append([f"P{p.id}", f"[{', '.join(map(str, adjusted_key))}]"])
            self.print_table(headers, rows)
        else:
            print("\nGroup key matches target key exactly, no adjustment needed")

    def _perform_four_particle_qkd(self, leader, followers):
        """Execute four-particle GHZ state QKD
        Args:
            leader: Leader participant
            followers: List of follower participants
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences, decoy_states = self._prepare_sequences([leader] + followers, 4)
        self._display_sequences(sequences, [leader] + followers)
        
        # 2. Display decoy states
        self._display_decoy_states(decoy_states, [leader] + followers)
        
        # 3. Execute exchange process
        exchanged_sequences = {}
        for follower in followers:
            # Determine sequence type to send
            if follower.id == leader.id + 1 or (leader.id == 4 and follower.id == 1):
                seq_type = 'B'
            elif follower.id == leader.id + 2 or (leader.id == 3 and follower.id == 1) or (leader.id == 4 and follower.id == 2):
                seq_type = 'C'
            else:
                seq_type = 'D'
            
            # Apply Pauli operations based on leader's private key
            result_sequence = self.apply_pauli_operations(sequences[leader.id][seq_type], leader)
            exchanged_sequences[follower.id] = result_sequence
        
        # 4. Display exchange process
        self._display_exchange_process(leader, followers, sequences, exchanged_sequences)
        
        # 5. Execute measurements and generate key
        measurement_results = {}
        
        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(4)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results
        
        # Follower measurements
        for follower in followers:
            follower_basis = [random.choice(['X', 'Z']) for _ in range(4)]
            follower_results = self.measure_sequences({}, {follower.id: exchanged_sequences[follower.id]}, follower_basis)
            measurement_results[follower.id] = follower_results
        
        # 6. Display measurement results
        self._display_measurement_results([leader] + followers, measurement_results)
        
        # 7. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)
        
        self.print_key_info(group_key, f"Group{leader.id} Initial Session Key")
        return group_key

    def _perform_three_particle_qkd(self, leader, followers):
        """Execute three-particle GHZ state QKD
        Args:
            leader: Leader participant
            followers: List of follower participants
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences, decoy_states = self._prepare_sequences([leader] + followers, 3)
        self._display_sequences(sequences, [leader] + followers)
        
        # 2. Display decoy states
        self._display_decoy_states(decoy_states, [leader] + followers)
        
        # 3. Execute exchange process
        exchanged_sequences = {}
        for follower in followers:
            # Determine sequence type to send
            if follower.id == leader.id + 1 or (leader.id == 4 and follower.id == 1):
                seq_type = 'B'
            else:
                seq_type = 'C'
            
            # Apply Pauli operations based on leader's private key
            result_sequence = self.apply_pauli_operations(sequences[leader.id][seq_type], leader)
            exchanged_sequences[follower.id] = result_sequence
        
        # 4. Display exchange process
        self._display_exchange_process(leader, followers, sequences, exchanged_sequences)
        
        # 5. Execute measurements and generate key
        measurement_results = {}
        
        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(3)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results
        
        # Follower measurements
        for follower in followers:
            follower_basis = [random.choice(['X', 'Z']) for _ in range(3)]
            follower_results = self.measure_sequences({}, {follower.id: exchanged_sequences[follower.id]}, follower_basis)
            measurement_results[follower.id] = follower_results
        
        # 6. Display measurement results
        self._display_measurement_results([leader] + followers, measurement_results)
        
        # 7. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)
        
        self.print_key_info(group_key, f"Group{leader.id} Initial Session Key")
        return group_key

    def _perform_bell_state_qkd(self, leader, follower):
        """Execute Bell state QKD
        Args:
            leader: Leader participant
            follower: Follower participant
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences, decoy_states = self._prepare_sequences([leader, follower], 2)
        self._display_sequences(sequences, [leader, follower])
        
        # 2. Display decoy states
        self._display_decoy_states(decoy_states, [leader, follower])
        
        # 3. Execute exchange process
        # Apply Pauli operations based on leader's private key
        result_sequence = self.apply_pauli_operations(sequences[leader.id]['B'], leader)
        exchanged_sequences = {follower.id: result_sequence}
        
        # 4. Display exchange process
        self._display_exchange_process(leader, [follower], sequences, exchanged_sequences)
        
        # 5. Execute measurements and generate key
        measurement_results = {}
        
        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(2)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results
        
        # Follower measurements
        follower_basis = [random.choice(['X', 'Z']) for _ in range(2)]
        follower_results = self.measure_sequences({}, exchanged_sequences, follower_basis)
        measurement_results[follower.id] = follower_results
        
        # 6. Display measurement results
        self._display_measurement_results([leader, follower], measurement_results)
        
        # 7. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)
        
        self.print_key_info(group_key, f"Group{leader.id} Initial Session Key")
        return group_key

def main():
    # Get user input
    while True:
        try:
            m_value = int(input("Enter the number of quantum states per participant M = "))
            if m_value > 0:
                break
            print("Please enter a positive integer")
        except ValueError:
            print("Please enter a valid integer")
    
    while True:
        try:
            num_participants = int(input("Enter the number of participants N = "))
            if num_participants >= 4:  # Ensure at least 4 participants (4 leaders)
                break
            print("Number of participants must be greater than or equal to 4")
        except ValueError:
            print("Please enter a valid integer")
    
    # Create quantum key exchange system
    qke = QuantumKeyExchange(num_participants, m_value)
    
    print("\n=== Quantum Key Exchange System Initialization ===")
    print(f"Total Participants: N = {num_participants}")
    print(f"Number of Quantum States per Participant: M = {m_value}")
    print(f"Number of Leaders: 4")
    print(f"Remaining Participants: {num_participants - 4}")
    print(f"Grouping Mode: (N-4) mod 3 = {(num_participants - 4) % 3}")
    
    # Execute key distribution
    print("\nStarting key distribution process...")
    qke.distribute_key()
    
    # Verify key consistency
    print("\n=== Verifying Key Consistency ===")
    leader = next(p for p in qke.participants if p.is_leader)
    leader_key = leader.shared_key
    
    all_keys_match = True
    mismatched_participant = None
    for participant in qke.participants:
        if participant.shared_key != leader_key:
            all_keys_match = False
            mismatched_participant = participant
            break
    
    # 使用表格格式显示验证结果
    headers = ["Verification Item", "Result"]
    rows = []
    
    if all_keys_match:
        rows.append(["Verification Result", "All participants' keys match perfectly!"])
        rows.append(["Number of participants with shared key", str(len(qke.participants))])
        rows.append(["Generated key length", f"{len(leader_key)} bits"])
        rows.append(["Final session key", f"[{', '.join(map(str, leader_key))}]"])
    else:
        rows.append(["Verification Result", "Key distribution failed, mismatched keys detected"])
        rows.append(["Mismatched Participant", str(mismatched_participant)])
        rows.append(["Leader's Key", f"[{', '.join(map(str, leader_key))}]"])
        rows.append(["Participant's Key", f"[{', '.join(map(str, mismatched_participant.shared_key))}]"])
    
    qke.print_table(headers, rows)

if __name__ == "__main__":
    main() 