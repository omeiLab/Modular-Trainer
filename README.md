# Modular-Trainer

**Modular-Trainer** is a clean, extensible training loop skeleton designed for machine learning experiments.  
The core idea is **modular hooks**: the training loop only controls the **time axis** (epoch/step), while hooks handle all events such as logging, checkpointing, and early stopping.

---

## Key Concepts

- **Training Loop**
  - Only manages the epoch and step iteration.
  - Does not know model, loss, optimizer, or metrics.
  - Calls hooks at appropriate times.

- **Hooks**
  - **before_epoch**: reset best scores, reset epoch statistics.
  - **after_step**: scheduler step, logging, checkpoint saving, early stopping checks.
  - **after_epoch**: display best scores, load best checkpoint for testing, epoch-level logging.

- **Control Object**
  - Determines whether to continue training, stop, or save checkpoint.
  - Loop only reacts to the control object’s decision.

---

## Repository Structure

```
modular-trainer/
├── src/
│ ├── trainer/      # loop skeleton + hook interfaces
│ ├── hooks/        # implementations for before_epoch, after_step, after_epoch
│ └── control/      # control logic (early stopping, checkpointing)
├── examples/       # dummy model / dataset usage
├── tests/          # loop skeleton tests
└── README.md
```


---

## Usage

1. Implement hooks according to your experiment needs.
2. Define a control object that decides continuation and checkpointing.
3. Initialize the training loop with your hooks and control object.
4. Run the loop; the hooks will handle all step and epoch events.

---

## Philosophy

- **Separation of Concerns**: Loop controls only the timing; all logic is in hooks.
- **Extensibility**: New hooks can be added without modifying the loop.
- **Reusability**: Same loop can be used for different models, datasets, or tasks.

---

> This is a skeleton for reproducible, modular, and clean ML experiments.
