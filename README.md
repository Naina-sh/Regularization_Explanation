# Regularization in Machine Learning

## 📌 Overview

Regularization is a technique used in Machine Learning to reduce **overfitting** and improve a model's ability to generalize to unseen data.

A model can sometimes become too complex and learn the training data too closely, including noise and irrelevant patterns. Regularization helps prevent this by adding a **penalty for large model weights** to the loss function.

Without regularization:

Loss = Prediction Error

With regularization:

Total Loss = Prediction Error + Regularization Penalty

The goal is to make the model fit the data well while avoiding unnecessarily large weights.

---

## 🎯 Why Do We Need Regularization?

A model can perform extremely well on training data but poorly on unseen/test data. This is called **overfitting**.

Regularization discourages the model from becoming unnecessarily complex by penalizing large weights.

The intuition is:

> "Fit the data well, but don't rely too heavily on large weights."

This helps the model learn more general patterns instead of memorizing the training data.

---

# 🧠 What is a Weight?

In a neural network, every connection between neurons has a weight.

For example:

Input x₁ ---- w₁ ----\
Input x₂ ---- w₂ ----- > Neuron
Input x₃ ---- w₃ ----/

A large weight means that the corresponding input has a stronger influence on the neuron's output.

Regularization discourages weights from becoming unnecessarily large.

---

# 🔢 General Regularization Formula

Without regularization:

Loss_total = Loss_original

With regularization:

Loss_total = Loss_original + λ × Penalty

Where:

- `Loss_original` = normal prediction loss
- `Penalty` = regularization penalty
- `λ (lambda)` = regularization strength

A larger λ means a stronger regularization penalty.

---

# 🟢 L2 Regularization

L2 regularization adds the **sum of squared weights** to the loss.

## Formula

Loss_total = Loss_original + λ × Σ(wᵢ²)

For example, suppose:

w₁ = 2
w₂ = 3

Then:

Σ(wᵢ²) = 2² + 3²
         = 4 + 9
         = 13

If:

λ = 0.05

Then:

Penalty = 0.05 × 13
        = 0.65

Therefore:

Loss_total = Loss_original + 0.65

The model now has an incentive to keep its weights smaller.

---

## 💡 Intuition Behind L2

L2 says:

> "Large weights are expensive, so try to keep the weights small."

For example:

Weight = 1

L2 penalty:

1² = 1

Weight = 10

L2 penalty:

10² = 100

Therefore, L2 penalizes large weights much more strongly.

L2 generally **shrinks weights toward zero**, but usually does not make them exactly zero.

This is why L2 is also commonly associated with **weight decay**.

---

# 🔵 L1 Regularization

L1 regularization adds the **sum of absolute values of the weights** to the loss.

## Formula

Loss_total = Loss_original + λ × Σ|wᵢ|

For example:

w₁ = 2
w₂ = -3

Then:

Σ|wᵢ| = |2| + |-3|
       = 2 + 3
       = 5

If:

λ = 0.05

Then:

Penalty = 0.05 × 5
        = 0.25

Therefore:

Loss_total = Loss_original + 0.25

---

## 💡 Intuition Behind L1

L1 says:

> "Try to keep weights small, and some weights may become exactly zero."

For example:

Before L1:

w = [0.8, 0.03, -0.7, 0.01, 1.2]

After L1:

w = [0.8, 0, -0.7, 0, 1.2]

Weights that become zero effectively stop contributing to the model.

Therefore, L1 can produce a **sparse model**.

---

# ⚖️ L1 vs L2

| Property | L1 Regularization | L2 Regularization |
|----------|-------------------|-------------------|
| Penalty | λ × Σ\|w\| | λ × Σw² |
| Effect on weights | Can make weights exactly 0 | Shrinks weights toward 0 |
| Sparse model | Yes | Usually No |
| Feature selection | Useful | Less directly |
| Large weights | Penalized | Strongly penalized |
| Common name | Lasso | Ridge |
| Keras | `l1()` | `l2()` |
---
