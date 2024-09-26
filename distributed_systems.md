# Distributed Systems

## First Lecture

DA RECUPERARE !!!

A Space-Time diagram shows the evolution of a distributed system over time,
it is composed of a number of timelines, each associated to a process $p_i$.
Events, occur across this timelines, and are represented by points on the diagram.
If an event $e_1$ is causally related to another event $e_2$, then $e_1$ is to the left of $e_2$ and there is a line connecting them.

## Second Lecture

### Definitions

We now give a set of formal definitions for very natural concepts in distributed systems.

We define the *history* of a process $p_i$ as the sequence of all events that occur in the timeline of $p_i$.

$$
h_i = \left\langle e^1_i, e^2_i, \ldots, e^n_i \right\rangle
$$

We can then have the history of the computation, which is a collection of all the histories of all the processes.

$$
H = \left\langle h_1, h_2, \ldots, h_n \right\rangle
$$

A *prefix history* is a sequence of events that occur in the timeline of a process up to a certain point.

$$
h_i^k = \left\langle e^1_i, e^2_i, \ldots, e^k_i \right\rangle
$$

A *local state* of process $i$ after event $e_i^k$ is the result of the computation up to that point, and is denoted as $\sigma_i^k$.

A *global state* is the collection of all local states of all processes at a certain point in time.

$$
\sigma^k = \left\langle \sigma_1^k, \sigma_2^k, \ldots, \sigma_n^k \right\rangle
$$

A *run* $k'$ is a total re-ordering of the events such that the internal order of every process is preserved. This means that a *run* allows the interleaving of the events of different processes, but does not break the order of events in the history of a single process.

A *cut* $C$ is a collection of the prefixes of every processes' history up to a certain point.

### Consistency

A cut $C$ is said to be *consistent* iff:

$$
e \rightarrow e^\prime \land e^\prime \in C \Rightarrow e \in C
$$

Or, in other words, if an event is in the cut, then all the events that causally precede it are also in the cut.

By sequentializing the events in a run and extracting a prefix, we can trivially obtain a consistent cut.

#### Consistency of Intersection

A possible course exercise is, given two cuts $C_1$ and $C_2$, to prove that their intersection is also a consistent cut.

Given the definition of consistency accomodated for the intersection of two cuts:

$$
e \rightarrow e^\prime \land e^\prime \in C_1 \cap C_2 \Rightarrow e \in C_1 \cap C_2
$$

We know that if $e^\prime$ is in the intersection, then it is in both $C_1$ and $C_2$. By the definition of consistency, we know that $e$ must be in both $C_1$ and $C_2$, and therefore in their intersection.

Formally:

$$
e \to e^\prime \land e^\prime \in C_1 \to e \in C_1
$$

$$
e \to e^\prime \land e^\prime \in C_2 \to e \in C_2
$$

Therefore:

$$
e \to e^\prime \land e^\prime \in C_1 \cap C_2 \to e \in C_1 \land e \in C_2 \to e \in C_1 \cap C_2
$$

#### Consistency of Union

Given two consistent cuts $C_1$ and $C_2$, we can prove that their union is also a consistent cut.

Given the definition of consistency accomodated for the union of two cuts:

$$
e \rightarrow e^\prime \land e^\prime \in C_1 \cup C_2 \Rightarrow e \in C_1 \cup C_2
$$

Then, formally:

$$
e \to e^\prime \land e^\prime \in C_1 \to e \in C_1
$$

$$
e \to e^\prime \land e^\prime \in C_2 \to e \in C_2
$$

Therefore:

$$
e \to e^\prime \land e^\prime \in C_1 \cup C_2 \to e \in C_1 \lor e \in C_2 \to e \in C_1 \cup C_2
$$

### Run Reconstruction

Assume that we have a set of processes $P = \left\{ p_1, p_2, p_3, \right\}$, and a monitor process $p_0$ that is responsible for detecting deadlocks.

Everytime that an event is generated, its responsible process sends a message to the monitor, which then updates its local state.

The monitor process can then reconstruct the run by aggregating the local process states into a global state.

Assuming asynchrony and unbounded message delays, the monitor process does not reconstruct a run, since the order of the individual processes can be inverted.

If we assume a FIFO message delivery, then the monitor process can reconstruct a run. This can be trivially achieved by having a sequence number, in a system such as TCP.

This does *not* recover consistency, since the order of separate processes can still be inverted. However, this level of *local consistency* is sufficient for deadlock detection.

#### Simplifying Assumptions for Consistency

Assume that I have:

1. A finite $\delta_i$ notification delay on each process $p_i$.
2. A global clock $C$ that is synchronized with all processes. We call an idealized version of this clock *Real Clock* (RC).


Then, the monitor process can offline-reconstruct a run by using the global clock to order the events. 

Online-reconstruction can be achieved (and is preferred) by exploiting the additional $\delta_i$ information. Once we receive a notification, we await time $\max \delta_i$ before committing the event to the global state, in the meantime, we store the event in a buffer that is ordered by the global clock.

#### Real Clock Property

The clock property relates the *happens before* relation of messages to the real time of the system.

$$
e \to e^\prime \Rightarrow TS(e) < TS(e^\prime)
$$

We can obtain a trivial clock by designing a system as such:

1. Each process has associates a sequence number $k \in \mathbb{N}$ to each event.
2. Each event $i$, which receives a message $m$ from event $j$, has a sequence number given by the formula $\max(k_i, k_j) + 1$.

This gives us the clock property *by construction*, since each event has a sequence number that is always greater than the sequence number of the event that caused it and 
of any event that it is casually related to.

This device is called a *Lamport Clock*, or a *Logical Clock*.

#### History of an event

Given an event $e$, we can define its history as the sequence of events that causally precede it. These are all the events that the monitor process must have received before the event $e$ is committed to the global state.

$$
\Theta(e) = \left\{ e^\prime \mid e^\prime \to e \right\}
$$

This is a *consistent* cut.

#### Opaque detection of consistent cuts

Given a notification about an event, for example, $e^4_5$, the monitor process can ensure that $e^4_1$ can be committed to global state by checking that for every process,
we have received a notification with lamport clock $\geq 4$.

We might have some processes that consistently lag behind, and therefore do not reach the lamport clock in a timely manner. We can trivially solve this by sending a liveness check to all processes for which a sufficiently high lamport number has not been received. If the process is alive but not working, it will re-adjust its lamport clock and send a notification.

This will not impact throughput, since the monitor process is not a bottleneck, and the liveness check is a very lightweight operation, that is performed only when necessary.

#### Building a Better Clock

We can build a *better clock*, that is, one with the enhanced property:

$$
e \to e^\prime \iff TS(e) < TS(e^\prime)
$$

This is called a *Vector Clock*.

It works by communicating the history of the current event in a compact way, we do this by sending a vector of length $n$ where $n$ is the number of processes in the system.
The vector's entries are updated to be the highest lamport number detected for that relative process, either from a message that has been received, or from the local lamport clock (in the case of the current process).

This allows us to check whether:

$$
VC(e) \subseteq VC(e^\prime)
$$

$$
e \to e^\prime \iff VC(e) \subseteq VC(e^\prime)
$$
When messages are sent, the vector clock is attached to the message, and the receiving process updates its vector clock by taking the maximum of the two vectors, and summing 1 to the current process.

The monitor process $p_0$ keeps track of its own vector, where each entry is the highest lamport number that it has received from each process. When it receives a notification, if the vector clock is consistent with the monitor's vector clock, then the event is committed to the global state.

Since vectors are *not* sets, we have to use component-wise comparison to check for causality.

$$
VC(e) < VC(e^\prime) \iff \forall i \in \left\{ 1, 2, \ldots, n \right\} \quad VC(e)_i < VC(e^\prime)_i
$$

So we are back to the original definition of the happens before relation.