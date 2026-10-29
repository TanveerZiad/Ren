# Distributed Systems Architecture

## Consensus and Raft
The Raft consensus algorithm is designed to be easily understandable while offering equivalent fault-tolerance to Paxos. It decomposes consensus into leader election, log replication, and safety invariants. Nodes in Raft can be in one of three states: Follower, Candidate, or Leader.

## CAP Theorem Trade-offs
Under network partitions, a distributed system must choose between consistency (every read receives the most recent write or an error) and availability (every non-failing node returns a response). Systems cannot achieve both consistency and availability simultaneously during arbitrary network failure partitions.

## Vector Clocks
Vector clocks determine partial ordering of events in distributed systems without synchronized physical clocks, detecting concurrent modifications and causality violations.

