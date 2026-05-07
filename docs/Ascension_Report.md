# Ascension Report: Project Genesis

## Exercise 1: The Legacy Chokehold (Sequential Simulation)

Die Simulation von 100.000 gedämpften harmonischen Oszillatoren mit Standard-NumPy und einer Python-Zeitschleife diente als Performance-Baseline.

- **Gesamtzeit (Legacy):** 0.8684 Sekunden
- **Beobachtung:** Obwohl NumPy die Array-Operationen effizient handhabt, verhindert der Overhead der Python-`for`-Schleife (1.000 Iterationen) eine maximale Auslastung der Hardware.

## Exercise 2: The Tensor Multiverse (vmap & jit)

In dieser Aufgabe wurde die Simulation nach JAX portiert, um moderne Accelerator-Features zu nutzen.

- **Status:** Implementiert in `src/jax_swarm.py`.
- **Technologien:** 
    - `vmap`: Zur automatischen Vektorisierung der Berechnungen über alle Oszillatoren hinweg.
    - `jit`: Zur Kompilierung der gesamten Zeitschleife via XLA (Accelerated Linear Algebra).
- **Messwerte (Lokal):** *Warten auf Ausführung*
- **Speedup Faktor:** *Warten auf Ausführung*

### Das Tracing-Phänomen
Der erste Aufruf einer mit `@jax.jit` dekorierten Funktion ist deutlich langsamer, da JAX die Funktion "traced", um einen optimierten Graphen für den XLA-Compiler zu erstellen. Erst der zweite Lauf profitiert von der direkten Ausführung des optimierten Maschinencodes auf dem Silicon.
