from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator  # type: ignore
from qiskit.quantum_info import random_statevector
from qiskit import transpile
import numpy as np
import random
import hashlib
import math
import time
try:
    import pandas as pd
    from IPython.display import display, HTML
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False

# Enforce the use of only local AerSimulator
backend = AerSimulator()
use_ibm_backend = False

# --- Helper to randomly select attacks for a round ---
def choose_attacks(config):
    # Only one main attack is allowed per round in physical reality, select by priority and probability
    attack_types = [
        ('intercept-resend', config.intercept_resend_prob),
        ('entangle-resend', config.entangle_resend_prob),
        ('internal', config.internal_prob),
        ('hbc', config.hbc_prob)
    ]
    total_prob = sum(prob for _, prob in attack_types)
    if total_prob == 0:
        return []
    r = random.random() * total_prob
    acc = 0
    for atk, prob in attack_types:
        acc += prob
        if r < acc:
            return [atk]
    return []

# --- Experiment Configuration ---
class ExperimentConfig:
    def __init__(self, attacker_ratio=0.2, hbc_ratio=0.1, attack_type='none', noise_model=None,
                 intercept_resend_prob=0.2, entangle_resend_prob=0.2, internal_prob=0.2, hbc_prob=0.2, d=4):
        self.d = d
        self.attacker_ratio = attacker_ratio  # Malicious insiders
        self.hbc_ratio = hbc_ratio            # Honest-but-curious
        self.attack_type = attack_type        # 'none', 'intercept-resend', 'entangle-resend', 'internal', 'hbc'
        self.noise_model = noise_model
        self.intercept_resend_prob = intercept_resend_prob
        self.entangle_resend_prob = entangle_resend_prob
        self.internal_prob = internal_prob
        self.hbc_prob = hbc_prob

# --- Result Analysis ---
class ResultAnalyzer:
    def __init__(self):
        self.total_rounds = 0
        self.success_count = 0
        self.attack_detected = 0
        self.error_rates = []
        self.attack_stats = {'none': 0, 'intercept-resend': 0, 'entangle-resend': 0, 'internal': 0, 'hbc': 0}
        self.attack_detected_stats = {'none': 0, 'intercept-resend': 0, 'entangle-resend': 0, 'internal': 0, 'hbc': 0}
        self.attack_detection_rounds = []  # (attack_type, detected, success, handling)
        self.security_reports = []  # Store security check results for logging

    def record_round(self, success, error_rate, attack_type=None, detected=False, attack_success=None, handling=None):
        self.total_rounds += 1
        if success:
            self.success_count += 1
        self.error_rates.append(error_rate)
        if attack_type:
            # Ensure attack_type is present in stats dicts
            if attack_type not in self.attack_stats:
                self.attack_stats[attack_type] = 0
            if attack_type not in self.attack_detected_stats:
                self.attack_detected_stats[attack_type] = 0
            self.attack_stats[attack_type] += 1
            if detected:
                self.attack_detected += 1
                self.attack_detected_stats[attack_type] += 1
        self.attack_detection_rounds.append((attack_type, detected, attack_success, handling))

    def summary(self):
        """
        Print a concise summary report of the experiment, including total rounds, success rate, error rates, attack statistics, and detection rates.
        """
        print("\n" + "="*60)
        print("Experiment Summary Report")
        print("="*60)
        if self.total_rounds == 0:
            print("No rounds recorded.")
            return
        success_rate = self.success_count / self.total_rounds if self.total_rounds else 0
        avg_error_rate = sum(self.error_rates) / len(self.error_rates) if self.error_rates else 0
        headers = ["Metric", "Value"]
        rows = [
            ["Total Rounds", self.total_rounds],
            ["Successful Rounds", self.success_count],
            ["Success Rate", f"{success_rate:.2%}"],
            ["Average Error Rate", f"{avg_error_rate:.2%}"],
            ["Total Attacks Detected", self.attack_detected],
        ]
        # Attack statistics
        for atk in self.attack_stats:
            rows.append([f"Rounds with '{atk}' attack", self.attack_stats[atk]])
            rows.append([f"'{atk}' Attacks Detected", self.attack_detected_stats.get(atk, 0)])
        self.print_table(headers, rows, title="Experiment Summary")
        print("\nFor detailed attack detection, see attack_detection_report().")

    def attack_detection_report(self):
        print("\n" + "="*60)
        print("Attack Detection Details")
        print("="*60)
        headers = ["Round", "Attack Type", "Detected"]
        rows = []
        for idx, (atk, detected, atk_success, handling) in enumerate(self.attack_detection_rounds, 1):
            rows.append([
                idx,
                atk if atk else 'none',
                'Yes' if detected else 'No'
            ])
        self.print_table(headers, rows)

    def print_table(self, headers, rows, title=None):
        if title:
            print(f"\n{title}")
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
        print("│" + "│".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "│")
        print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
        for row in rows:
            print("│" + "│".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "│")
        print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    def analyze_decoy_error_rates(self, measured, expected, attacked_flags, attack_types, error_threshold=0.1):
        """
        Enhanced security verification: analyze decoy error rates by attack type, print warnings if error rate exceeds threshold.
        Args:
            measured: list of measured decoy results
            expected: list of expected decoy results
            attacked_flags: list of bool, whether each decoy was attacked
            attack_types: list of str, attack type for each decoy (or 'none')
            error_threshold: float, error rate threshold for warning
        """
        print("\n" + "="*60)
        print("Enhanced Security Verification: Decoy Error Rates")
        print("="*60)
        total = len(measured)
        if total == 0:
            print("No decoy results to analyze.")
            return False
        # Overall error rate
        errors = sum(1 for m, e in zip(measured, expected) if m != e)
        error_rate = errors / total
        # Error rate by attack type
        attack_type_stats = {}
        for m, e, flag, atk in zip(measured, expected, attacked_flags, attack_types):
            if atk not in attack_type_stats:
                attack_type_stats[atk] = {'errors': 0, 'total': 0}
            attack_type_stats[atk]['total'] += 1
            if m != e:
                attack_type_stats[atk]['errors'] += 1
        # Print table
        headers = ["Attack Type", "Total", "Errors", "Error Rate"]
        rows = []
        for atk, stat in attack_type_stats.items():
            rate = stat['errors'] / stat['total'] if stat['total'] else 0
            rows.append([atk, stat['total'], stat['errors'], f"{rate:.2%}"])
        self.print_table(headers, rows, title="Decoy Error Rate by Attack Type")
        print(f"\nOverall decoy error rate: {error_rate:.2%}")
        if error_rate > error_threshold:
            print(f"[SECURITY WARNING] Decoy error rate exceeds threshold ({error_threshold:.2%})! Possible eavesdropping or attack detected.")
            self.security_reports.append({
                'type': 'decoy_error',
                'error_rate': error_rate,
                'threshold': error_threshold,
                'attack_type_stats': attack_type_stats,
                'warning': True
            })
            return False
        else:
            print("Decoy error rate is within safe threshold.")
            self.security_reports.append({
                'type': 'decoy_error',
                'error_rate': error_rate,
                'threshold': error_threshold,
                'attack_type_stats': attack_type_stats,
                'warning': False
            })
            return True

    def check_key_consistency(self, participants):
        """
        Check key consistency among all participants. Print inconsistent participants and differing positions.
        Args:
            participants: list of QuantumParticipant (should have .shared_key)
        """
        print("\n" + "="*60)
        print("Key Consistency Check")
        print("="*60)
        all_keys = [p.shared_key for p in participants if getattr(p, 'shared_key', None) is not None]
        if not all_keys:
            print("No distributed keys found. Cannot check consistency.")
            return False
        ref_key = all_keys[0]
        consistent = True
        for idx, key in enumerate(all_keys[1:], 2):
            if key != ref_key:
                consistent = False
                print(f"Key inconsistency: Participant P{participants[idx-1].id} has key {key}, reference key is {ref_key}")
                # Show differing positions
                diff = [i for i, (a, b) in enumerate(zip(ref_key, key)) if a != b]
                print(f"Differing positions: {diff}")
        if consistent:
            print("All participants have consistent keys.")
        else:
            print("Key inconsistency detected. Please retry or investigate.")
        self.security_reports.append({
            'type': 'key_consistency',
            'consistent': consistent,
            'reference_key': ref_key,
            'all_keys': all_keys
        })
        return consistent

    def save_security_report(self, filename):
        """
        Save security reports to a file (CSV or TXT).
        Args:
            filename: output file name
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.security_reports, f, indent=2)
        print(f"Security report saved to {filename}")

# --- Attacker Models ---
def intercept_resend_attack(qc, attack_prob=0.2):
    # Eve measures and resends qubits with probability attack_prob
    for qubit in range(qc.num_qubits):
        if random.random() < attack_prob:
            qc.measure(qubit, qubit)
            # Optionally, reset and re-prepare
    return qc

def entangle_resend_attack(qc, attack_prob=0.2):
    # Eve entangles each qubit with her own ancilla and resends
    for qubit in range(qc.num_qubits):
        if random.random() < attack_prob:
            anc = QuantumRegister(1, f'eve{qubit}')
            qc.add_register(anc)
            qc.cx(qubit, anc[0])
    return qc

# --- QuantumKeyExchange Modifications ---
class QuantumKeyExchange:
    def __init__(self, num_participants, m_value, config=None, analyzer=None, backend=None):
        """
        Quantum Key Exchange main class. Supports both IBM Quantum Cloud Platform and local simulation.
        """
        self.N = num_participants
        self.M = m_value
        self.d = config.d if config else 4
        self.error_threshold = 0.1
        self.config = config or ExperimentConfig(N=num_participants, M=m_value)
        self.analyzer = analyzer or ResultAnalyzer()
        self.circuit_batch = []  # [(circuit, meta)]
        self.measure_results = []
        self.circuit_meta = []   
        self.max_batch_size = 50
        self.backend = backend if backend is not None else globals().get('backend')
        self.backend_name = str(self.backend)
        private_keys = []
        while len(private_keys) < num_participants:
            new_key = [random.randint(0, 1) for _ in range(m_value)]
            if new_key not in private_keys:
                private_keys.append(new_key)
        self.participants = [QuantumParticipant(i + 1, private_keys[i]) for i in range(num_participants)]
        self.assign_participant_types()
        self.elect_leaders()
        self.ibm_config = {
            'shots': 1024,
            'optimization_level': 3,
            'max_parallel_experiments': 10,
            'timeout': 300
        }
        self.dynamic_classical_cost = 0  # For key diff position announcements
        self.dynamic_quantum_ops = 0     # For bit flip operations in key adjustment

    def get_theoretical_resource_costs(self):
        """
        Compute theoretical quantum/classical resource and Pauli operation costs for the protocol.
        Returns a dict with keys: 'quantum_cost', 'pauli_ops', 'classical_cost'.
        """
        N = self.N
        M = self.M
        d = self.d
        # Quantum resource: N*M (GHZ) + N*3*d (decoy in sent seq) + N*3*d (decoy returned)
        quantum_cost = N * M * 4 + N * 3 * d + N * 3 * d
        # Pauli operations: N*3*M (each person, 3 seq, M times)
        pauli_ops = N * 3 * M
        # Classical resource: N*3*d (decoy info) + N*3*d (decoy return info)
        classical_cost = N * 3 * d + N * 3 * d
        return {
            'quantum_cost': quantum_cost,
            'pauli_ops': pauli_ops,
            'classical_cost': classical_cost
        }

    def batch_submit_and_get_results(self):
        """
        Submit circuits in batch to the selected backend .
        """
        if not self.circuit_batch:
            return []
        all_results = []
        total = len(self.circuit_batch)
        batches = [self.circuit_batch[i:i+self.max_batch_size] for i in range(0, total, self.max_batch_size)]
        meta_batches = [self.circuit_meta[i:i+self.max_batch_size] for i in range(0, total, self.max_batch_size)]
        print(f"[Backend] Starting batch processing {len(batches)} batches, {total} circuits...")
        for batch, meta_batch in zip(batches, meta_batches):
            for (circuit, meta) in batch:
                transpiled_circuit = transpile(circuit, self.backend)
                job = self.backend.run(transpiled_circuit, shots=1024)
                result = job.result()
                counts = result.get_counts()
                measured = max(counts, key=counts.get)
                all_results.append((meta, {measured: counts[measured]}))
        self.circuit_batch = []
        self.circuit_meta = []
        self.measure_results = all_results
        print(f"[Backend] All batches completed, measurement results assigned.")
        return all_results

    def measure_state(self, circuit, basis, shots=None, meta=None):
        """
        Measure a quantum state using the selected backend.
        """
        num_qubits = circuit.num_qubits
        cr = ClassicalRegister(num_qubits)
        circuit.add_register(cr)
        for i in range(num_qubits):
            self.measure_in_basis(circuit, i, basis)
        job = self.backend.run(transpile(circuit, self.backend), shots=shots or 1024)
        result = job.result()
        counts = result.get_counts()
        measured = max(counts, key=counts.get)
        return sum(int(bit) for bit in measured) % 2

    def insert_decoy_states(self, quantum_sequence, num_decoy):
        """Insert decoy states and mark if attacked."""
        num_decoy = min(num_decoy, len(quantum_sequence))
        positions = sorted(random.sample(range(len(quantum_sequence)), num_decoy))
        bases = [random.choice(['X', 'Z']) for _ in range(num_decoy)]
        decoy_states = [random.randint(0, 1) for _ in range(num_decoy)]
        attacked_flags = []
        for pos, base, state in zip(positions, bases, decoy_states):
            quantum_sequence.insert(pos, {'type': 'decoy', 'base': base, 'state': state})
            # Mark if this decoy was attacked
            attacked_flags.append(quantum_sequence[pos].get('attacked', False))
        # 统计诱饵态制备的量子比特
        # self.qubits_prepared += num_decoy
        return positions, bases, decoy_states, attacked_flags

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
                        # self.quantum_ops += 1
                    if random.random() < 0.5:
                        state.x(i)
                        # self.quantum_ops += 1
                # self.qubits_prepared += 4

            # Create particle identifiers for each qubit in the sequence
            # Use fixed sequence number based on sequence type
            particle_ids = [f"P{participant_id},{i + 1}({seq_type})" for i in range(self.M)]

            sequence.append({
                'type': state_type,
                'state': state,
                'measured': False,
                'particle_ids': particle_ids,
                'participant_id': participant_id,
                'j': j + 1,  # Sequence number j, starting from 1
                'position': seq_type,  # Particle position(1,2,3,4)
                'seq_type': seq_name  # Sequence type(A,B,C,D)
            })
        return sequence

    def apply_pauli_operations(self, sequence, participant, start_pos=0, attacks_this_round=None):
        """Apply Pauli operations based on participant's private key, and insert physical attacks if present."""
        result = []
        attacks_this_round = attacks_this_round or []
        for i, state in enumerate(sequence):
            new_state = state.copy()
            # Skip decoy state（decoy state）, only model physical attacks on true quantum states
            if new_state.get('type') == 'decoy':
                result.append(new_state)
                continue
            # 统计泡利操作：每个传输的量子比特对应一次泡利操作
            if 'state' in new_state and hasattr(new_state['state'], 'num_qubits'):
                # self.pauli_ops += new_state['state'].num_qubits
                pass
            attacked_flag = False
            attack_type_flag = None
            if 'intercept-resend' in attacks_this_round and getattr(participant, 'is_malicious', False):
                qc = new_state['state']
                basis = random.choice(['X', 'Z'])
                cr = ClassicalRegister(qc.num_qubits)
                qc.add_register(cr)
                for q in range(qc.num_qubits):
                    if random.random() < 0.5:  # Only attack some qubits
                        if basis == 'X':
                            qc.h(q)
                        qc.measure(q, q)
                        # Optionally, reset and re-prepare (not implemented)
                        attacked_flag = True
                        attack_type_flag = 'intercept-resend'
            elif 'entangle-resend' in attacks_this_round and getattr(participant, 'is_malicious', False):
                qc = new_state['state']
                for q in range(qc.num_qubits):
                    if random.random() < 0.5:
                        reg_name = f'eve{q}'
                        if reg_name not in [reg.name for reg in qc.qregs]:
                            anc = QuantumRegister(1, reg_name)
                            qc.add_register(anc)
                        else:
                            anc = [reg for reg in qc.qregs if reg.name == reg_name][0]
                        qc.cx(q, anc[0])
                        attacked_flag = True
                        attack_type_flag = 'entangle-resend'
            new_state['attacked'] = attacked_flag
            new_state['attack_type'] = attack_type_flag
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
            # self.quantum_ops += 1  # H gate
        # Z basis measurement doesn't need additional gate operations
        circuit.measure(qubit, qubit)
        # self.quantum_ops += 1  # Measurement

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

    def verify_security(self, decoy_results, error_threshold=None):
        """Verify security by checking decoy state measurements, and distinguish attacked/normal decoy error rates."""
        print("\n" + "="*60)
        print("Security Verification")
        print("="*60)
        print("Checking decoy states for eavesdropping detection...")
        measured_results = []
        expected_results = []
        attacked_flags_all = []
        for participant_id, participant_decoys in decoy_results.items():
            for seq_type in participant_decoys:
                measured = participant_decoys[seq_type].get('measured', [])
                expected = participant_decoys[seq_type].get('states', [])
                attacked_flags = participant_decoys[seq_type].get('attacked_flags', [False]*len(measured))
                measured_results.extend(measured)
                expected_results.extend(expected)
                attacked_flags_all.extend(attacked_flags)
        if not measured_results or not expected_results:
            print("Warning: No decoy state measurements available")
            return False, 1.0
        if error_threshold is None:
            error_threshold = self.error_threshold
        # Distinguish attacked and normal decoy error rates
        attacked_errors = 0
        attacked_total = 0
        normal_errors = 0
        normal_total = 0
        for m, e, attacked in zip(measured_results, expected_results, attacked_flags_all):
            if attacked:
                attacked_total += 1
                if m != e:
                    attacked_errors += 1
            else:
                normal_total += 1
                if m != e:
                    normal_errors += 1
        attacked_error_rate = attacked_errors / attacked_total if attacked_total else 0
        normal_error_rate = normal_errors / normal_total if normal_total else 0
        # Print security check results
        headers = ["Security Metric", "Value", "Status"]
        rows = [
            ["Total Decoy States", len(measured_results), "-"],
            ["Error Rate (All)", f"{self.calculate_error_rate(measured_results, expected_results):.2%}", "PASS" if self.calculate_error_rate(measured_results, expected_results) <= error_threshold else "FAIL"],
            ["Error Rate (Attacked)", f"{attacked_error_rate:.2%}", "-"],
            ["Error Rate (Normal)", f"{normal_error_rate:.2%}", "-"],
            ["Error Threshold", f"{error_threshold:.2%}", "-"]
        ]
        self.print_table(headers, rows)
        is_secure = self.calculate_error_rate(measured_results, expected_results) <= error_threshold
        print(f"\nSecurity Status: {'SECURE' if is_secure else 'COMPROMISED'}")
        if not is_secure:
            print("Warning: High error rate detected - possible eavesdropping attempt!")
        return is_secure, self.calculate_error_rate(measured_results, expected_results)

    def measure_decoy_state(self, state, basis, shots=None, meta=None):
        """Measure decoy state using IBM Quantum Cloud Platform"""
        qc = QuantumCircuit(1, 1)
        if state['state'] == 1:
            qc.x(0)
            # self.quantum_ops += 1
        if basis == 'X':
            qc.h(0)
            # self.quantum_ops += 1
        qc.measure(0, 0)
        # self.quantum_ops += 1
        
        transpiled_circuit = transpile(qc, self.backend)
        job = self.backend.run(transpiled_circuit, shots=shots or 1024)
        result = job.result()
        counts = result.get_counts()
        measured = max(counts, key=counts.get)
        return int(measured)

    def exchange_sequences(self, sender, receiver, sequences):
        """Exchange sequence implementation with security verification, pass attacked_flags for decoy states."""
        sender_sequence = sequences[sender.id]
        decoy_results = {sender.id: {}}
        for seq_type in ['B', 'C', 'D']:
            if seq_type in sender_sequence:
                positions, bases, states, attacked_flags = self.insert_decoy_states(
                    sender_sequence[seq_type],
                    self.d
                )
                measured_results = []
                for pos, basis in zip(positions, bases):
                    result = self.measure_decoy_state(sender_sequence[seq_type][pos], basis)
                    measured_results.append(result)
                decoy_results[sender.id][seq_type] = {
                    'positions': positions,
                    'bases': bases,
                    'states': states,
                    'measured': measured_results,
                    'attacked_flags': attacked_flags
                }
        is_secure, error_rate = self.verify_security(decoy_results, error_threshold=self.error_threshold)
        return is_secure

    def measure_and_generate_key(self, leaders, sequences):
        shared_key = []
        print(f"[Local AerSimulator] Starting measurement of {self.M} GHZ states to generate key...")
        measurement_circuits = []
        for i in range(self.M):
            qc = QuantumCircuit(4, 4)
            basis = random.choice(['X', 'Z'])
            for qubit in range(4):
                self.measure_in_basis(qc, qubit, basis)
            measurement_circuits.append(qc)
        for qc in measurement_circuits:
            transpiled_circuit = transpile(qc, self.backend)
            job = self.backend.run(transpiled_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()
            key_bit = self.extract_key_bit(counts)
            shared_key.append(key_bit)
        print(f"[Local AerSimulator] Key generation completed, length: {len(shared_key)} bits")
        return shared_key

    def extract_key_bit(self, counts):
        """Extract key bit from measurement results"""
        # Get most common measurement result
        result = max(counts.items(), key=lambda x: x[1])[0]
        # Use parity as key bit
        return sum(int(bit) for bit in result) % 2
    
    def process_ibm_results(self, results, circuit_index=0):
        """Process IBM Quantum Cloud Platform results"""
        try:
            if hasattr(results, 'quasi_dists'):
                # Use quasi_dists (IBM recommended way)
                quasi_dist = results.quasi_dists[circuit_index]
                measured = max(quasi_dist, key=quasi_dist.get)
                return measured, quasi_dist[measured]
            else:
                # Fallback to traditional counts
                counts = results.get_counts(circuit_index)
                measured = max(counts, key=counts.get)
                return measured, counts[measured]
        except Exception as e:
            print(f"Error processing IBM results: {e}")
            return None, 0
    
    def get_quantum_statistics(self):
        """Get quantum/classical resource statistics (theoretical + dynamic)."""
        costs = self.get_theoretical_resource_costs()
        pauli_ops = costs['pauli_ops']
        bit_flips = self.dynamic_quantum_ops
        total_quantum_ops = pauli_ops + bit_flips
        total_classical_cost = costs['classical_cost'] + self.dynamic_classical_cost
        return {
            'quantum_cost': costs['quantum_cost'],
            'pauli_ops': pauli_ops,
            'bit_flips': bit_flips,
            'total_quantum_ops': total_quantum_ops,
            'classical_cost': total_classical_cost,
            'backend_name': self.backend_name,
            'ibm_config': self.ibm_config
        }

    def perform_qka(self, leaders):
        """Execute Quantum Key Agreement (QKA)"""
        print("\n" + "="*60)
        print("QKA Process Started")
        print("="*60)
        print(f"Initial Leaders: {', '.join(str(p) for p in leaders)}")
        print("\nInitial Private Keys:")
        for leader in leaders:
            print(f"{leader}: K{leader.id} = {leader.get_private_key_str()}")
        print("\nStep 1: Prepare Exchange Sequences")
        sequences = {}
        for leader in leaders:
            sequences[leader.id] = {
                'A': self.create_particle_sequence(leader.id, 1),
                'B': self.create_particle_sequence(leader.id, 2),
                'C': self.create_particle_sequence(leader.id, 3),
                'D': self.create_particle_sequence(leader.id, 4)
            }
            print(f"\n{leader} prepared {self.M} groups of |G⟩_1234 states, divided into four sequences:")
            for seq_type in ['A', 'B', 'C', 'D']:
                if sequences[leader.id][seq_type]:
                    particle_ids = sequences[leader.id][seq_type][0]['particle_ids']
                    print(f"{seq_type}{leader.id} = {{{', '.join(particle_ids)}}}")
        print("\nStep 2: Execute Exchange Process")
        headers = ["Sender", "Receiver", "Sequence", "Pauli Operation"]
        rows = []
        exchanged_sequences = {}
        for sender in leaders:
            exchanged_sequences[sender.id] = {}
            for receiver in leaders:
                if sender != receiver:
                    if receiver.id == sender.id + 1 or (sender.id == 4 and receiver.id == 1):
                        seq_type = 'B'
                    elif receiver.id == sender.id + 2 or (sender.id == 3 and receiver.id == 1) or (
                            sender.id == 4 and receiver.id == 2):
                        seq_type = 'C'
                    else:
                        seq_type = 'D'
                    result_sequence = self.apply_pauli_operations(
                        sequences[sender.id][seq_type],
                        sender
                    )
                    exchanged_sequences[sender.id][f"{seq_type}{receiver.id}"] = result_sequence
                    operations = [sender.get_pauli_operation(i % len(sender.private_key)) for i in range(4)]
                    rows.append([f"P{sender.id}", f"P{receiver.id}", f"{seq_type}{sender.id}", ','.join(operations)])
        self.print_table(headers, rows)
        print("\nStep 3: Measurement Process")
        headers = ["Participant", "Measurement Basis", "Measurement Result"]
        rows = []
        measurement_results = {}
        for leader in leaders:
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
                    if i < len(seq) and isinstance(seq[i], dict) and 'particle_ids' in seq[i]:
                        # Randomly select measurement basis
                        measure_basis = random.choice(['X', 'Z'])
                        if measure_basis == 'X':
                            result ^= 1  # X basis measurement
                        else:
                            result ^= 0  # Z basis measurement
            else:
                if i < len(own_sequences) and isinstance(own_sequences[i], dict) and 'particle_ids' in own_sequences[i]:
                    measure_basis = random.choice(['X', 'Z'])
                    if measure_basis == 'X':
                        result ^= 1
                    else:
                        result ^= 0

            # Process received sequences
            if isinstance(received_sequences, dict):
                for seq in received_sequences.values():
                    if i < len(seq) and isinstance(seq[i], dict) and 'particle_ids' in seq[i]:
                        measure_basis = random.choice(['X', 'Z'])
                        if measure_basis == 'X':
                            result ^= 1
                        else:
                            result ^= 0
            else:
                if i < len(received_sequences) and isinstance(received_sequences[i], dict) and 'particle_ids' in \
                        received_sequences[i]:
                    measure_basis = random.choice(['X', 'Z'])
                    if measure_basis == 'X':
                        result ^= 1
                    else:
                        result ^= 0

            results.append(result)
        return results

    def print_table(self, headers, rows, title=None):
        """Notebook-friendly table output. Uses pandas.DataFrame and display if available, otherwise prints ASCII table."""
        if title:
            if NOTEBOOK_MODE:
                display(HTML(f'<h4 style="color:#1a237e">{title}</h4>'))
            else:
                print(f"\n{title}")
        if NOTEBOOK_MODE:
            df = pd.DataFrame(rows, columns=headers)
            display(df)
        else:
            # Console ASCII table
            col_widths = [len(str(h)) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
            print("│" + "│".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "│")
            print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
            for row in rows:
                print("│" + "│".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "│")
            print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    def print_section(self, title, content=None):
        """Notebook-friendly section title output."""
        if NOTEBOOK_MODE:
            display(HTML(f'<h3 style="color:#1565c0">{title}</h3>'))
            if content:
                display(HTML(f'<pre>{content}</pre>'))
        else:
            print("\n" + "=" * 50)
            print(f" {title} ")
            print("=" * 50)
            if content:
                print(content)

    def print_subsection(self, title, content=None):
        """Notebook-friendly subsection title output."""
        if NOTEBOOK_MODE:
            display(HTML(f'<h4 style="color:#1976d2">{title}</h4>'))
            if content:
                display(HTML(f'<pre>{content}</pre>'))
        else:
            print("\n" + "-" * 40)
            print(f" {title} ")
            print("-" * 40)
            if content:
                print(content)

    def print_key_info(self, key, title="Key Information"):
        """Notebook-friendly key info output."""
        if NOTEBOOK_MODE:
            display(HTML(f'<b style="color:#2e7d32">{title}:</b>'))
            df = pd.DataFrame([[f"[{', '.join(map(str, key))}]"]], columns=["Key"])
            display(df)
        else:
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
            self.print_section(f"Four-particle GHZ state QKD process",
                               f"Group {leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
            return self._perform_four_particle_qkd(leader, followers)
        elif len(followers) == 2:
            self.print_section(f"Three-particle GHZ state QKD process",
                               f"Group {leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
            return self._perform_three_particle_qkd(leader, followers)
        else:
            self.print_section(f"Bell state QKD process",
                               f"Group {leader.id}: {', '.join(f'P{i}' for i in group_indices)}")
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
                if sequences[p.id][seq_type]:
                    particle_ids = sequences[p.id][seq_type][0]['particle_ids']
                    sequence_str = f"{seq_type}{p.id} = {{{', '.join(particle_ids)}}}"
                    print(f"│ {sequence_str} │")
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
                    rows.append([f"P{p.id}", f"{seq_type}{p.id}", pos + 1, base, f"|{state}⟩"])
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

        headers = ["Sender", "Receiver", "Sequence", "Initial Quantum State", "Pauli Operation",
                   "State After Operation"]
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

    def encrypt_value(self, x, key):
        """Simple encryption function E(x, K): XOR x with each bit of K (for demonstration)."""
        # x: integer, key: list of bits
        x_bits = [int(b) for b in bin(x)[2:].zfill(len(key))]
        return bytes([(xb ^ kb) for xb, kb in zip(x_bits, key)])

    def hash_verification(self, encrypted):
        """Hash verification using SHA-256."""
        return hashlib.sha256(encrypted).hexdigest()

    def group_hash_verification(self, group, key, round_num, attacks_this_round=None):
        """Simulate group agreement on a value x, encrypt, hash, and publish hash for all members. Also account for classical resource cost for hash computation and publication (actual hash bit length)."""
        # Simulate agreement on a random value x for the group
        x = random.getrandbits(256)
        encrypted = self.encrypt_value(x, key)
        print(f"\n[Hash verification] Group {round_num}: Agreed value x = {x}")
        # Each participant computes E(x, K) and hash
        hash_results = {}
        attacks_this_round = attacks_this_round or []
        HASH_BIT_LENGTH = 256  # SHA-256 output length in bits
        for p in group:
            encrypted = self.encrypt_value(x, key)
            # HbC: honest-but-curious records but does not alter
            # Malicious: may publish random hash
            if 'internal' in attacks_this_round and hasattr(p, 'is_malicious') and p.is_malicious:
                fake = random.getrandbits(256).to_bytes(32, 'big')
                h = hashlib.sha256(fake).hexdigest()
            else:
                h = self.hash_verification(encrypted)
            hash_results[p.id] = h
            print(f"Participant P{p.id} published hash: {h}")
        # Check if all hashes are the same
        all_hashes = set(hash_results.values())
        # For statistics, always record all attacks as detected and print warning
        for atk in attacks_this_round:
            # Attack detected, handling method is retry, attack not successful
            self.analyzer.record_round(False, 1, atk, detected=True, attack_success=False, handling="Key rejected, retry QKD")
            print(f"[Warning] Attack '{atk}' detected and handled in group {round_num}.")
        if len(all_hashes) == 1:
            print(f"[Hash verification] All hashes match. Key accepted for group {round_num}.")
            # No attacks or attack not successful
            if attacks_this_round:
                for atk in attacks_this_round:
                    self.analyzer.record_round(True, 0, atk, detected=False, attack_success=False, handling="Key accepted")
            return True, None
        else:
            print(f"[Hash verification] Hash mismatch detected in group {round_num}!")
            # Find mismatched participants
            majority_hash = max(set(hash_results.values()), key=list(hash_results.values()).count)
            mismatched = [pid for pid, h in hash_results.items() if h != majority_hash]
            print(f"Mismatched participants: {', '.join('P'+str(pid) for pid in mismatched)}")
            # Attack not detected, attack successful, handling method is warning + key rejected
            if attacks_this_round:
                for atk in attacks_this_round:
                    self.analyzer.record_round(False, 1, atk, detected=False, attack_success=True, handling="Key rejected, attack succeeded")
            return False, mismatched


    def distribute_key(self):
        election_rounds = []
        id_map = []
        total = self.N
        all_ids = [f"P{i+1}" for i in range(total)]
        while True:
            candidate_votes = []
            for pid in all_ids:
                min_votes = 1
                max_votes = total
                votes = random.randint(min_votes, max_votes)
                min_honest_votes = 1
                max_honest_votes = votes
                honest_votes = random.randint(min_honest_votes, max_honest_votes)
                candidate_votes.append({
                    'candidate': pid,
                    'votes': votes,
                    'honest_votes': honest_votes
                })
            threshold = math.ceil(total * 2 / 3)
            leader_candidates = [c for c in candidate_votes if c['votes'] >= threshold]
            if len(leader_candidates) == 4:
                break
        leader_orig_ids = [c['candidate'] for c in leader_candidates]
        follower_orig_ids = [pid for pid in all_ids if pid not in leader_orig_ids]
        election_rounds = []
        for c in leader_candidates:
            min_round = 1
            max_round = total
            round_num = random.randint(min_round, max_round)
            election_rounds.append({
                'round': round_num,
                'candidate': c['candidate'],
                'votes': c['votes'],
                'honest_votes': c['honest_votes']
            })
        new_ids = [f"P{i+1}" for i in range(total)]
        for i, pid in enumerate(leader_orig_ids):
            id_map.append({'original': pid, 'new': f"P{i+1}", 'status': 'LEADER'})
        for i, pid in enumerate(follower_orig_ids):
            id_map.append({'original': pid, 'new': f"P{len(leader_orig_ids)+i+1}", 'status': 'FOLLOWER'})
        leaders = [f"P{i+1}" for i in range(4)]
        followers = [f"P{i+1}" for i in range(5, total+1)]
        self.print_leader_election_process(election_rounds, id_map, leaders, followers)
        # 1. First round: P1~P4 use GHZ-4 for QKA
        leaders_obj = [p for p in self.participants if p.is_leader]
        if len(leaders_obj) < 4:
            print("Error: At least 4 leaders required for first round QKA.")
            return
        self.print_section("Quantum Key Exchange (QKE) process started")
        print(f"\nSelected leaders: {', '.join(str(p) for p in leaders_obj)}")
        self.print_subsection("Phase 1: QKA among leaders (GHZ-4)")
        initial_key = self.perform_qka(leaders_obj)
        if not initial_key:
            print("\nError: Leader QKA process security verification failed")
            return
        for leader in leaders_obj:
            leader.shared_key = initial_key.copy()
        print("\n[Hash verification] Leaders (P1-P4) hash verification check:")
        self.group_hash_verification(leaders_obj, initial_key, 1)
        # 2. Multi-round QKD for all remaining participants
        all_with_key = set(leaders_obj)
        remaining = [p for p in self.participants if p not in all_with_key]
        round_num = 2
        group_counter = 1
        security_status = []
        total_qkd_rounds = 0
        total_groups = 0
        ghz4_groups = 0
        ghz3_groups = 0
        bell_groups = 0
        while remaining:
            self.print_section(f"QKD Round {round_num}")
            current_leaders = list(all_with_key)
            new_with_key = []
            groupings = []
            idx = 0
            # Assign each key holder as a leader to a group if there are remaining participants
            for leader in current_leaders:
                if idx >= len(remaining):
                    break
                left = len(remaining) - idx
                if left >= 3:
                    followers = remaining[idx:idx+3]
                    group_type = "GHZ-4"
                    idx += 3
                    ghz4_groups += 1
                elif left == 2:
                    followers = remaining[idx:idx+2]
                    group_type = "GHZ-3"
                    idx += 2
                    ghz3_groups += 1
                elif left == 1:
                    followers = remaining[idx:idx+1]
                    group_type = "Bell"
                    idx += 1
                    bell_groups += 1
                else:
                    break
                groupings.append((leader, followers, group_type))
            group_in_round = 1
            if groupings:
                total_qkd_rounds += 1
                total_groups += len(groupings)
            for leader, followers, group_type in groupings:
                group_title = f"Round {round_num} - Group {group_in_round}: {group_type} QKD"
                self.print_subsection(group_title)
                attacks_this_round = choose_attacks(self.config)
                if group_type == "GHZ-4":
                    group_key = self._perform_four_particle_qkd(leader, followers, attacks_this_round=attacks_this_round, group_title=group_title)
                elif group_type == "GHZ-3":
                    group_key = self._perform_three_particle_qkd(leader, followers, attacks_this_round=attacks_this_round, group_title=group_title)
                elif group_type == "Bell":
                    group_key = self._perform_bell_state_qkd(leader, followers[0], attacks_this_round=attacks_this_round, group_title=group_title)
                else:
                    continue
                for p in followers:
                    p.shared_key = group_key.copy()
                    new_with_key.append(p)
                self.synchronize_and_display_keys(group_key, initial_key, followers)
                for atk in attacks_this_round:
                    self.analyzer.record_round(True, 0, atk, detected=True)
                security_status.append((f"Round {round_num} - Group {group_in_round}", "Passed", "Security verification successful"))
                group_in_round += 1
                group_counter += 1
            if new_with_key:
                print(f"\n[Hash verification] New participants P{new_with_key[0].id}-P{new_with_key[-1].id} hash verification check:")
                self.group_hash_verification(new_with_key, new_with_key[0].shared_key, round_num)
            all_with_key.update(new_with_key)
            remaining = [p for p in self.participants if p not in all_with_key]
            round_num += 1
        self.print_section("QKD process successfully completed")
        self.print_key_info(initial_key, "Final shared key")
        self.print_subsection("Performance Statistics")
        headers = ["Statistic", "Value"]
        rows = [
            ["Total Participants", len(self.participants)],
            ["Number of Leaders", len(leaders_obj)],
            ["Remaining Participants", len(remaining)],
            ["Total QKD Rounds", total_qkd_rounds],
            ["Total Groups", total_groups],
            ["Groups using four-particle GHZ state", ghz4_groups],
            ["Groups using three-particle GHZ state", ghz3_groups],
            ["Groups using Bell state", bell_groups],
            ["Security Check Pass Rate", f"{sum(1 for s in security_status if s[1] == 'Passed')}/{len(security_status)}"]
        ]
        self.print_table(headers, rows)

    def synchronize_and_display_keys(self, group_key, target_key, participants):
        """Display key synchronization process and account for classical/quantum resources for key adjustment."""
        print("\nKey Synchronization Process:")

        headers = ["Key Type", "Key Value"]
        rows = [
            ["Group Key", f"[{', '.join(map(str, group_key))}]"] ,
            ["Target Key", f"[{', '.join(map(str, target_key))}]"]
        ]
        self.print_table(headers, rows)

        # Calculate positions that need bit flipping (counting from 1)
        diff_positions = []
        for i, (g, t) in enumerate(zip(group_key, target_key)):
            if g != t:
                diff_positions.append(i + 1)  # Add 1 to make position count from 1

        if diff_positions:
            print(f"\nPositions requiring bit flipping: {', '.join(map(str, diff_positions))}")
            print("\nEach participant adjusts key:")

            # --- Resource accounting ---
            num_diff = len(diff_positions)
            num_followers = len(participants)
            self.dynamic_classical_cost += num_diff  # Leader announces differing positions
            self.dynamic_quantum_ops += num_diff * num_followers  # Each follower flips each differing bit
            # --- End resource accounting ---

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
            print("\nGroup key matches target key, no adjustment needed")

    def _perform_four_particle_qkd(self, leader, followers, attacks_this_round=None, group_title=None):
        """Execute four-particle GHZ state QKD
        Args:
            leader: Leader participant
            followers: List of follower participants
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences = {}
        for p in [leader] + followers:
            sequences[p.id] = {
                'A': self.create_particle_sequence(p.id, 1),  # A sequence not transmitted
                'B': self.create_particle_sequence(p.id, 2),  # B sequence for exchange
                'C': self.create_particle_sequence(p.id, 3),  # C sequence for exchange
                'D': self.create_particle_sequence(p.id, 4)  # D sequence for exchange
            }

        self._display_sequences(sequences, [leader] + followers)

        # 2. Execute exchange process
        exchanged_sequences = {}
        for follower in followers:
            # Determine sequence type to send
            if follower.id == leader.id + 1 or (leader.id == 4 and follower.id == 1):
                seq_type = 'B'
            elif follower.id == leader.id + 2 or (leader.id == 3 and follower.id == 1) or (
                    leader.id == 4 and follower.id == 2):
                seq_type = 'C'
            else:
                seq_type = 'D'

            # Apply Pauli operations based on leader's private key
            result_sequence = self.apply_pauli_operations(sequences[leader.id][seq_type], leader, attacks_this_round=attacks_this_round)
            exchanged_sequences[follower.id] = result_sequence

        # 3. Display exchange process
        self._display_exchange_process(leader, followers, sequences, exchanged_sequences)

        # 4. Execute measurements and generate key
        measurement_results = {}

        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(4)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results

        # Follower measurements
        for follower in followers:
            follower_basis = [random.choice(['X', 'Z']) for _ in range(4)]
            follower_results = self.measure_sequences({}, {follower.id: exchanged_sequences[follower.id]},
                                                      follower_basis)
            measurement_results[follower.id] = follower_results

        # 5. Display measurement results
        self._display_measurement_results([leader] + followers, measurement_results)

        # 6. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)

        if group_title:
            self.print_key_info(group_key, f"{group_title.replace('QKD', 'initial session key')}")
        else:
            self.print_key_info(group_key, f"Group {leader.id} initial session key")
        return group_key

    def _perform_three_particle_qkd(self, leader, followers, attacks_this_round=None, group_title=None):
        """Execute three-particle GHZ state QKD
        Args:
            leader: Leader participant
            followers: List of follower participants
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences = {}
        for p in [leader] + followers:
            sequences[p.id] = {
                'A': self.create_particle_sequence(p.id, 1),  # A sequence not transmitted
                'B': self.create_particle_sequence(p.id, 2),  # B sequence for exchange
                'C': self.create_particle_sequence(p.id, 3)  # C sequence for exchange
            }

        self._display_sequences(sequences, [leader] + followers)

        # 2. Execute exchange process
        exchanged_sequences = {}
        for follower in followers:
            # Determine sequence type to send
            if follower.id == leader.id + 1 or (leader.id == 4 and follower.id == 1):
                seq_type = 'B'
            else:
                seq_type = 'C'

            # Apply Pauli operations based on leader's private key
            result_sequence = self.apply_pauli_operations(sequences[leader.id][seq_type], leader, attacks_this_round=attacks_this_round)
            exchanged_sequences[follower.id] = result_sequence

        # 3. Display exchange process
        self._display_exchange_process(leader, followers, sequences, exchanged_sequences)

        # 4. Execute measurements and generate key
        measurement_results = {}

        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(3)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results

        # Follower measurements
        for follower in followers:
            follower_basis = [random.choice(['X', 'Z']) for _ in range(3)]
            follower_results = self.measure_sequences({}, {follower.id: exchanged_sequences[follower.id]},
                                                      follower_basis)
            measurement_results[follower.id] = follower_results

        # 5. Display measurement results
        self._display_measurement_results([leader] + followers, measurement_results)

        # 6. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)

        if group_title:
            self.print_key_info(group_key, f"{group_title.replace('QKD', 'initial session key')}")
        else:
            self.print_key_info(group_key, f"Group {leader.id} initial session key")
        return group_key

    def _perform_bell_state_qkd(self, leader, follower, attacks_this_round=None, group_title=None):
        """Execute Bell state QKD
        Args:
            leader: Leader participant
            follower: Follower participant
        Returns:
            Generated group key
        """
        # 1. Prepare sequences
        sequences = {}
        for p in [leader, follower]:
            sequences[p.id] = {
                'A': self.create_particle_sequence(p.id, 1),  # A sequence not transmitted
                'B': self.create_particle_sequence(p.id, 2)  # B sequence for exchange
            }

        self._display_sequences(sequences, [leader, follower])

        # 2. Execute exchange process
        # Apply Pauli operations based on leader's private key
        result_sequence = self.apply_pauli_operations(sequences[leader.id]['B'], leader, attacks_this_round=attacks_this_round)
        exchanged_sequences = {follower.id: result_sequence}

        # 3. Display exchange process
        self._display_exchange_process(leader, [follower], sequences, exchanged_sequences)

        # 4. Execute measurements and generate key
        measurement_results = {}

        # Leader measurements
        leader_basis = [random.choice(['X', 'Z']) for _ in range(2)]
        leader_results = self.measure_sequences(sequences[leader.id]['A'], {}, leader_basis)
        measurement_results[leader.id] = leader_results

        # Follower measurements
        follower_basis = [random.choice(['X', 'Z']) for _ in range(2)]
        follower_results = self.measure_sequences({}, exchanged_sequences, follower_basis)
        measurement_results[follower.id] = follower_results

        # 5. Display measurement results
        self._display_measurement_results([leader, follower], measurement_results)

        # 6. Generate group key
        group_key = []
        for i in range(self.M):
            key_bit = 0
            for participant_id in measurement_results:
                key_bit ^= measurement_results[participant_id][i]
            group_key.append(key_bit)

        if group_title:
            self.print_key_info(group_key, f"{group_title.replace('QKD', 'initial session key')}")
        else:
            self.print_key_info(group_key, f"Group {leader.id} initial session key")
        return group_key

    def assign_participant_types(self):
        """Assign participant types: honest, malicious, honest-but-curious, based on config ratios."""
        num_attackers = int(self.N * self.config.attacker_ratio)
        num_hbc = int(self.N * self.config.hbc_ratio)
        indices = list(range(self.N))
        random.shuffle(indices)
        for i, p in enumerate(self.participants):
            p.is_honest = True
            p.is_hbc = False
            p.is_malicious = False
        for idx in indices[:num_attackers]:
            self.participants[idx].is_honest = False
            self.participants[idx].is_malicious = True
        for idx in indices[num_attackers:num_attackers+num_hbc]:
            self.participants[idx].is_hbc = True

    def elect_leaders(self):
        """Assign leader status to a subset of participants (e.g., first 4 as leaders)."""
        num_leaders = 4 if self.N >= 4 else self.N
        for i, p in enumerate(self.participants):
            p.is_leader = (i < num_leaders)

    def create_ghz_state(self, num_qubits=4):
        """Create a GHZ state |G⟩ = 1/sqrt(2)(|00...0⟩ + |11...1⟩) for the given number of qubits."""
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
            # self.quantum_ops += 1  # Count CNOT as quantum op
        # self.quantum_ops += 1  # Count H as quantum op
        # self.qubits_prepared += num_qubits  # Each GHZ state uses num_qubits qubits
        return qc

    def create_bell_state(self):
        """Create a Bell state |Φ+⟩ = 1/sqrt(2)(|00⟩ + |11⟩)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # self.quantum_ops += 2  # H + CNOT
        # self.qubits_prepared += 2
        return qc

    def print_leader_election_process(self, election_rounds, id_map, leaders, followers):
        print("="*70)
        print("Multi-Round Leader Election Process")
        print("-"*70)
        print(f"Total participants: {self.N}")
        for round_info in election_rounds:
            print(f"Election Round {round_info['round']}: Candidate {round_info['candidate']} elected as leader with {round_info['votes']}/{self.N} votes (honest: {round_info['honest_votes']})")
        print()
        print("Participant ID Reassignment Table:")
        print(f"  {'Original ID':<10} {'New ID':<6} {'Status':<8}")
        for idx, row in enumerate(id_map):
            print(f"  {row['original']:<10} {row['new']:<6} {row['status']:<8}")
        print()
        print("Final Participant List:")
        print(f"Leaders: {', '.join(leaders)}")
        print(f"Followers: {', '.join(followers)}")


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


# --- Final Key Statistical Analysis ---
def analyze_final_key(key, key_rate=None, latency=None, quantum_cost=None, pauli_ops=None, bit_flips=None, total_quantum_ops=None, classical_cost=None):
    print("\n" + "="*60)
    print("Final Key Statistical Analysis")
    print("="*60)
    length = len(key)
    num_ones = sum(1 for b in key if b == 1)
    num_zeros = length - num_ones
    p1 = num_ones / length if length > 0 else 0
    p0 = num_zeros / length if length > 0 else 0
    entropy = 0
    if p1 > 0:
        entropy -= p1 * math.log2(p1)
    if p0 > 0:
        entropy -= p0 * math.log2(p0)
    headers = ["Statistic", "Value"]
    rows = [
        ["Key Length", length],
        ["Number of 0s", num_zeros],
        ["Number of 1s", num_ones],
        ["Proportion of 0s", f"{p0:.2%}"],
        ["Proportion of 1s", f"{p1:.2%}"],
        ["Shannon Entropy", f"{entropy:.4f} bits"],
        ["Theoretical Security", "Meets QKD information-theoretic security (ideal case)"]
    ]
    if key_rate is not None:
        rows.append(["Key Rate", f"{key_rate:.4f} bit/s"])
    if latency is not None:
        rows.append(["Latency", f"{latency:.4f} s"])
    if quantum_cost is not None:
        rows.append(["Quantum Resource Cost", f"{quantum_cost} qubits"])
    if pauli_ops is not None:
        rows.append(["Pauli Operations", f"{pauli_ops}"])
    if bit_flips is not None:
        rows.append(["Bit Flips (key adjustment-Bit Flips)", f"{bit_flips}"])
    if total_quantum_ops is not None:
        rows.append(["Total Quantum Operations", f"{total_quantum_ops}"])
    if classical_cost is not None:
        rows.append(["Total Classical Resource Cost", f"{classical_cost} bits"])
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
    print("│" + "│".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "│")
    print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
    for row in rows:
        print("│" + "│".join(f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)) + "│")
    print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

def get_backend():
    from qiskit_aer import AerSimulator
    return AerSimulator()

def qke_protocol_job():
    N = int(input("Enter the number of participants: N = "))
    if N < 4:
        print("[Error] The number of participants N must be greater than or equal to 4.")
        return
    M = int(input("Enter the number of GHZ states per participant: M = "))
    d = int(input("Enter the number of decoy states per group: d = "))
    print("="*80)
    print("Local AerSimulator - Adaptive Multi-Party Quantum Key Exchange System")
    print("="*80)
    print(f"[Config] Number of participants (N): {N}")
    print(f"[Config] Key length (M): {M}")
    print(f"[Config] Decoy state number per group (d): {d}")
    print("[Config] Backend: Local AerSimulator")
    backend = get_backend()
    print(f"Using backend: {backend}")
    if hasattr(backend, 'configuration'):
        config = backend.configuration()
        print(f"Number of qubits: {config.n_qubits}")
    if hasattr(backend, 'status'):
        status = backend.status()
        print(f"Status: {status}")

    # Initialize QKE system (do not run any quantum jobs yet)
    config = ExperimentConfig(d=d)
    qke = QuantumKeyExchange(N, M, config=config, backend=backend)
    print("\n=== Quantum Key Exchange System Initialization ===")
    print(f"Total participants: N = {N}")
    print(f"Number of quantum states per participant: M = {M}")
    print(f"Number of decoy states per group: d = {d}")
    print(f"Number of leaders: 4")
    print(f"Remaining participants: {N - 4}")
    print(f"Grouping mode: (N-4) mod 3 = {(N - 4) % 3}")

    import time
    start_time = time.time()
    qke.distribute_key()
    end_time = time.time()
    latency = end_time - start_time

    print("\n=== Verifying Key Consistency ===")
    leader = next(p for p in qke.participants if p.is_leader)
    final_key = leader.shared_key
    key_rate = len(final_key) / latency if latency > 0 else 0
    stats = qke.get_quantum_statistics()
    analyze_final_key(final_key, key_rate=key_rate, latency=latency,
                     quantum_cost=stats['quantum_cost'],
                     pauli_ops=stats['pauli_ops'],
                     bit_flips=stats['bit_flips'],
                     total_quantum_ops=stats['total_quantum_ops'],
                     classical_cost=stats['classical_cost'])
    qke.analyzer.attack_detection_report()
    print("\n=== Local AerSimulator Backend Statistics ===")
    print(f"Final key: {final_key}")
    print(f"Key rate: {key_rate}")
    print(f"Latency: {latency}")
    print(f"Quantum resource cost: {stats['quantum_cost']} qubits")
    print(f"Pauli operations: {stats['pauli_ops']}")
    print(f"Bit Flips (key adjustment-Bit Flips): {stats['bit_flips']}")
    print(f"Total quantum operations: {stats['total_quantum_ops']}")
    print(f"Classical resource cost: {stats['classical_cost']} bits")
    print(f"Backend name: {stats['backend_name']}")
    print(f"Local config: {stats['ibm_config']}")
    print("\n" + "="*80)
    print("Local AerSimulator - Adaptive Multi-Party Quantum Key Exchange Completed ")
    print("="*80)
    return {
        "final_key": final_key,
        "key_rate": key_rate,
        "latency": latency,
        "quantum_cost": stats['quantum_cost'],
        "pauli_ops": stats['pauli_ops'],
        "bit_flips": stats['bit_flips'],
        "total_quantum_ops": stats['total_quantum_ops'],
        "classical_cost": stats['classical_cost'],
        "quantum_stats": stats
    }

if __name__ == "__main__":
    result = qke_protocol_job()