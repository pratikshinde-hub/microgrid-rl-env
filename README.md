---
title: Microgrid RL Environment
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
---

# ⚡ Microgrid RL Environment

A production-grade reinforcement learning environment simulating a real-world microgrid system.

An agent must optimally control battery usage and flexible loads under uncertainty to minimize cost and avoid blackouts.

---

## 🧠 Problem

Modern energy systems must handle:

- intermittent solar generation  
- fluctuating electricity prices  
- uncertain demand  
- possible grid outages  

This environment captures those challenges in a controllable RL setting.

---

## 🎯 Objective

The agent must:

- minimize energy cost  
- prevent blackouts  
- maximize solar utilization  
- manage deferred (flexible) load  
- maintain battery health  

---

## ⚙️ Environment Overview

- Episode length: **96 steps (15-minute intervals)**
- Deterministic with fixed seed
- Fully reproducible

### Observation Space

Includes:

- solar generation & forecast  
- battery state-of-charge (SOC)  
- base + flexible load  
- electricity price  
- grid availability  
- timestep information  

### Action Space

| Action | Description |
|------|-------------|
| `battery_kw` | Charge (+) or discharge (-) battery |
| `curtail_fraction` | Fraction of flexible load to defer |

---

## 🧪 Tasks

| Task | Difficulty | Description |
|------|----------|------------|
| sunny_day | Easy | Stable solar & demand |
| volatile_market | Medium | High price volatility |
| islanded_crisis | Hard | Grid outage scenario |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|--------|--------|------------|
| `/reset` | GET / POST | Start new episode |
| `/step` | GET / POST | Apply action |
| `/state/{id}` | GET | Get current state |
| `/tasks` | GET | List tasks |
| `/grader` | GET / POST | Score episode |
| `/baseline` | GET / POST | Heuristic action |
| `/baseline_run` | GET | Full baseline evaluation |
| `/health` | GET | Service status |

---

## 🏆 Grading System

Final score ∈ **[0, 1]** based on:

- cost efficiency  
- blackout avoidance  
- solar self-consumption  
- deferred load clearance  
- battery health  

---

## 🤖 Baseline Agent

A heuristic policy that:

- charges when prices are low  
- discharges when prices are high  
- avoids extreme battery usage  
- prioritizes blackout prevention  

Use `/baseline_run` to evaluate full performance.

---

## 🔁 Determinism

- Same task + seed → identical results  
- Fully reproducible evaluation  

---

## 🚀 Quick Start

### Reset
```bash
POST /reset
{
  "task_id": "sunny_day",
  "seed": 42
}