# AIC Solution

Our solution for Intrinsic's [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge):
a robot arm autonomously plugging SFP and SC network connectors into their
ports.

Over two rounds we built and trained two complete, very different insertion
pipelines. For Qualification, we had no given port detection, so we labeled
our own dataset and trained a YOLO keypoint model to find the ports, wrote
the multi-camera triangulation to turn those detections into a 3D pose, and
trained a vision-based regressor (on data collected with our own capture
pipeline) to correct residual pose error before insertion. For Phase 1, port
detection moved to Intrinsic's FlowState, so we reworked the policy around
force control instead — detecting contact, telling a clean insertion apart
from an edge catch, and recovering from snags — while carrying the
offset-correction model over unchanged. Against teams of up to ten, the two
of us placed 27th of 160 in Qualification and 14th of 160 overall. The
sections below give the short version; the linked docs go into the actual
pipelines, math, and tuning for anyone who wants the details.

> **Note:** This repo contains only our solution code. For the full toolkit
> and setup instructions, follow the official
> [getting started guide](https://github.com/intrinsic-dev/aic/blob/main/docs/getting_started.md)
> first, then [docs/environment-setup.md](docs/environment-setup.md) for how
> this package plugs into it.

---

## Results

| Round | Approach | Result |
|---|---|---|
| Qualification | Own vision pipeline (YOLO port detection + triangulation) | 27 / 160 teams advanced |
| Phase 1 | Intrinsic FlowState for perception, force-controlled insertion | **14 / 160 teams** |

## Demo

Phase-1 policy plugging in both connector types:

**SC insertion**

<video src="https://github.com/user-attachments/assets/24a9969e-ccff-41e1-bcb1-2e3e374ae856" width="240" controls></video>

**SFP insertion**

<video src="https://github.com/user-attachments/assets/372e5b8e-c86d-4b76-b7c8-ba8532941530" width="480" controls></video>

**Full insertion process** (both connectors, end to end)

<video src="https://github.com/user-attachments/assets/f5bd5886-9559-4a56-973c-9471196d7263" width="480" controls></video>

Qualification-round port detection (YOLO keypoints, triangulated into a 3D pose) — *screenshot coming soon*.

---

## The two policies

The task board has two connector types (SFP, SC). Both rounds needed a
policy that could find the port, get the plug aligned, and actually seat it
under contact — but *how* it found the port changed completely between
rounds, so each round got its own dedicated insertion policy in
[`aic_solution_policy`](aic_solution_policy/):

- **[Qualification round](docs/qualification-phase.md)** –
  [`QualificationPlugIn`](aic_solution_policy/aic_solution_policy/ros/qualification/QualificationPlugIn.py):
  a custom-trained YOLO keypoint model finds the port corners in all three
  camera views, triangulated into a 3D port pose, then a spiral search
  handles the final alignment.
- **[Phase 1](docs/phase1-flowstate.md)** –
  [`Phase1PlugIn`](aic_solution_policy/aic_solution_policy/ros/phase1/Phase1PlugIn.py):
  port detection moved into Intrinsic FlowState, so the policy became
  perception-free and purely force-controlled — descend to contact, tell a
  clean insertion apart from an edge catch, and recover from snags.

Both stages share a small vision model that nudges the approach pose from
camera images (see below).

## Docs

| Doc | What's in it |
|---|---|
| [qualification-phase.md](docs/qualification-phase.md) | How the YOLO keypoint model + multi-camera triangulation find the port, pipeline diagram |
| [phase1-flowstate.md](docs/phase1-flowstate.md) | The FlowState hand-off, the force-controlled descend/contact/press state machine, snag recovery |
| [residual-offset-correction.md](docs/residual-offset-correction.md) | The offset-correction regressor shared by both policies, and how its training data was collected |
| [environment-setup.md](docs/environment-setup.md) | pixi/distrobox setup, starting the sim, teleop |
| [running-and-testing.md](docs/running-and-testing.md) | Step-by-step commands to run/debug each policy and to collect training data |
| [aic_solution_policy/README.md](aic_solution_policy/README.md) | Package layout and dev workflow (reinstall-after-change, etc.) |
| [training/README.md](training/README.md) | Reading the YOLO keypoint model's training metrics |
