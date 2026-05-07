# Ascension Report: Project Genesis

## Exercise 1: The Legacy Chokehold (Sequential Simulation)

Die Simulation von 100.000 gedämpften harmonischen Oszillatoren mit Standard-NumPy und einer Python-Zeitschleife diente als Performance-Baseline.

- **Gesamtzeit (Legacy):** 1.3432 Sekunden
- **Beobachtung:** Obwohl NumPy die Array-Operationen effizient handhabt, verhindert der Overhead der Python-`for`-Schleife (1.000 Iterationen) eine maximale Auslastung der Hardware.

## Exercise 2: The Tensor Multiverse (vmap & jit)

In dieser Aufgabe wurde die Simulation nach JAX portiert, um moderne Accelerator-Features zu nutzen.

- **Status:** Implementiert in `src/jax_swarm.py`.
- **Technologien:** 
    - `vmap`: Zur automatischen Vektorisierung der Berechnungen über alle Oszillatoren hinweg.
    - `jit`: Zur Kompilierung der gesamten Zeitschleife via XLA (Accelerated Linear Algebra).
- **Messwerte (Lokal):** *Warten auf Ausführung*
- **Speedup Faktor:** *Warten auf Ausführung*

## Exercise 3: Time Travel via Gradients (grad)

In dieser Aufgabe wurde die Simulation als differenzierbarer mathematischer Graph behandelt. Statt Trial-and-Error wurde der optimale Parameter (Anfangsgeschwindigkeit) direkt über Gradienten bestimmt.

- **Status:** Implementiert in `src/jax_optimization.py`.
- **Ziel:** Erreichen einer Distanz von 150.0m nach 5s.
- **Ergebnis:** *Warten auf Ausführung (Optimierte Geschwindigkeit)*

## Exercise 4: Agentic Refactoring for the Horizon (Flax)

In der finalen Aufgabe wurde der Übergang von zustandsorientierten (Object-Oriented) Architekturen wie Keras hin zu zustandslosen (stateless) Modellen mit Flax vollzogen.

- **Status:** Implementiert in `src/flax_core.py`.
- **Modell:** Multi-Layer Perceptron (MLP) mit `flax.linen`.

### Explizites vs. Implizites State-Management
In klassischen Frameworks wie Keras sind die Gewichte Teil des Modell-Objekts (`model.weights`). Dies erzeugt "Side-Effects", die JAX-Transformationen wie `jit` oder `vmap` erschweren. Flax trennt die **Architektur** (das `nn.Module`) strikt von den **Parametern** (dem `params` Dictionary). 

Die Initialisierung erfolgt über `model.init`, welches die Gewichte zurückgibt, ohne sie im Modell zu speichern. Der Forward Pass wird über `model.apply` ausgeführt, wobei die Gewichte explizit als Argument übergeben werden müssen. Diese funktionale Reinheit ist der Schlüssel für die Skalierbarkeit und Differenzierbarkeit in modernen KI-Systemen.
