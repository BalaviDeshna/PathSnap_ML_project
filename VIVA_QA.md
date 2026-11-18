# PathSnap — Presentation & Viva Preparation

## 60-second project explanation

“PathSnap solves GPS map matching: noisy GPS observations are aligned to a directed road network to reconstruct the route travelled. For each GPS observation, we generate nearby road-segment candidates and calculate spatial, heading, speed and timing features. We compare six classifiers using trip-grouped cross-validation, then optimise Random Forest hyperparameters using Gaussian-process Bayesian optimisation. Candidate probabilities are passed to a connectivity-aware Viterbi decoder to produce a coherent road sequence. FastAPI serves predictions and a Leaflet UI visualises the GPS trace and reconstructed routes.”

## Key viva questions

**1. What is map matching?**  
Converting noisy GPS observations into the most likely sequence of road segments on a digital road network.

**2. Why is GPS alone insufficient?**  
GPS can drift because of multipath effects, buildings, signal interference, dropouts and sampling gaps. A point may appear on a nearby parallel road.

**3. What do the four input files contain?**  
`.track`: longitude, latitude and timestamp. `.nodes`: road-node coordinates. `.arcs`: directed connections between node IDs. `.route`: chronological reference arc IDs.

**4. What is an arc?**  
A directed road segment represented by a start-node ID and an end-node ID. Its row index in `.arcs` is its arc ID.

**5. Why generate candidates?**  
We compare plausible nearby road segments instead of every road in the network, reducing the search space.

**6. What features do you use?**  
Point-to-segment distance, midpoint distance, segment length, heading difference, GPS speed, candidate rank, distance to both endpoints and timestamp gap.

**7. Which six models did you compare?**  
Logistic Regression, KNN, Random Forest, Extra Trees, HistGradientBoosting and AdaBoost.

**8. Why GroupKFold?**  
Rows from the same trip are correlated. GroupKFold keeps all candidate rows from a trip in one fold, reducing leakage between training and validation.

**9. What is Macro F1 here?**  
The unweighted mean of F1 for the two binary candidate classes (candidate is reference-route match or not). It gives both classes equal weight.

**10. Why did Random Forest perform well?**  
It captures nonlinear feature interactions and gave the highest candidate-level grouped-CV Macro F1 among the six classifiers in this experiment.

**11. What is candidate miss rate?**  
The percentage of GPS observations for which the reference road segment is absent from the generated candidate set. A classifier cannot predict a segment that was never generated.

**12. What is HMM?**  
A Hidden Markov Model treats the true road segment as a hidden state and GPS-derived classifier scores as evidence (emissions). It uses road connectivity to model plausible transitions.

**13. What does Viterbi do?**  
It uses dynamic programming to find the highest-scoring sequence of hidden states rather than selecting the best candidate independently at every observation.

**14. How is transition connectivity checked?**  
Arc A can directly precede arc B when A's end node equals B's start node. Staying on the same arc is also allowed. Disconnected transitions receive a penalty.

**15. Did HMM help?**  
For the Bayesian-tuned Random Forest on the 10 student-test trips, mean per-trip Top-1 increased from about 73.78% to 75.22%; mean route continuity increased from about 86.48% to 97.26%.

**16. What is Bayesian optimisation?**  
A search strategy using a probabilistic surrogate model (here a Gaussian Process) to select promising hyperparameter configurations using expected improvement.

**17. What was the best Bayesian trial?**  
78 trees, maximum depth 19 and minimum leaf size 2, with candidate Macro F1 about 0.8503 on the optimisation subset.

**18. What are the main limitations?**  
Fixed top-8 candidate generation, approximate point labels derived from the route sequence, a fixed disconnected-transition penalty, and simplified handling of travel-time constraints. The decoder restarts after timestamp gaps over 30 seconds.

**19. How do you prevent leakage?**  
Training and CV use the 80 training trips. The 10 student-test trips are used only for development comparisons. The reserved evaluator partition is separate and is not included in this package.

**20. What does the API return?**  
It returns the GPS trace, classifier route geometry and Viterbi route geometry. It does not return reference labels or metrics calculated from them.

**21. Why two laptops?**  
Laptop 1 hosts the model and API. Laptop 2 hosts the evaluator-facing UI and sends requests to Laptop 1 over the local network.

## Current measured numbers to remember

- 80 training trips; 10 student-testing trips.
- 208,417 training GPS observations; 19,160 student-test observations.
- 6 classifiers compared with 3-fold GroupKFold. The final KNN model is fitted on a deterministic 15,000-row subset for practical inference.
- Random Forest CV candidate Macro F1: **0.8373**.
- Best Bayesian trial: **78 trees, depth 19, min leaf 2; CV Macro F1 0.8503** on the optimisation subset.
- Bayesian-tuned RF student-test candidate Macro F1: **0.8515**.
- Bayesian-tuned RF student-test observation Top-1: **73.78% classifier-only; 75.22% after Viterbi**.
- Bayesian-tuned RF route continuity: **86.48% classifier-only; 97.26% after Viterbi**.
- Student-test candidate miss rate: **6.28%**.

Be precise: candidate-level Macro F1, observation-level Top-1 accuracy and route continuity are different metrics. The student-test metrics are development results, not evaluator results.
