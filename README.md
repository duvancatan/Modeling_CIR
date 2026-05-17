# Modeling Transmission Intensity in SI Epidemics via CIR and Jacobi Processes

This repository contains the numerical implementation and simulation scripts for the paper:  

**"Modeling Transmission Intensity in SI Epidemics via CIR and Jacobi
Processes: Asymptotic Results and Preliminary Intervention Strategies"**.

## Authors

* **León A. Valencia** (Corresponding Author: lalexander.valencia@udea.edu.co)

* **Raúl Alejandro Morán-Vásquez**

* **Duván H. Cataño Salazar**

*Instituto de Matemáticas, Universidad de Antioquia, Medellín, Colombia*.

## Abstract

This project introduces a stochastic framework for modeling epidemic transmission rates using processes of the form $\beta_{t}=\varphi(t)P_{t}$. We use the **Cox-Ingersoll-Ross (CIR)** and the **Jacobi** process to capture the intrinsic randomness of infections $P_t$, while a deterministic function $\varphi(t)$ models the impact of public health interventions. 

Key contributions include:

* **Asymptotic Analysis:** We prove that a susceptible fraction of the population survives ($S_{\infty} > 0$) if and only if the integrated intensity process $H_t = \int_0^t \beta_s ds$ remains bounded as $t \to \infty$.

* **Risk Assessment:** Derivation of a "risk monitor" based on Chernoff bounds to estimate the probability of reaching critical infection thresholds.

* **Process Comparison:** Analysis of how bounded (Jacobi) vs. unbounded (CIR) stochastic drivers affect epidemic saturation and risk estimation.

## Project Structure

The code is organized to reproduce the figures and results presented in the paper:

* `src/`: Python scripts for SDE simulations using the Euler-Maruyama scheme.

* `figures/`: Directory where generated plots are saved automatically.

## Installation & Requirements

The simulations are implemented in ```Python 3.8+```. To install the necessary dependencies, run:


```bash
pip install -r requirements.txt
```

