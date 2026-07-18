# SfpPlugInPhase1 – Testen & Debuggen

Perception-freie SFP-Insert-Policy: keine Kameras/YOLO, keine Port-Erkennung.
Sie geht davon aus, dass der Roboter schon korrekt über dem Ziel-Port steht
und der Stecker gegriffen/ausgerichtet ist. Zielpose = Startpose (einmalig
abgefragt) minus `_insertion_offset_z`.

Policy-Datei: `aic_solution_policy/aic_solution_policy/SfpPlugInPhase1.py`
Debug-Positionierungs-Skript: `aic_bringup/scripts/position_over_port_debug.py`

Voraussetzung für das Debug-Skript: Simulation mit **Ground Truth TF**
(`ground_truth:=true`), da es TF-Frames wie `..._entrance` und
`cable_0/sfp_tip_link` braucht, die es sonst nicht gibt.

---

## 1. Kompletter Testablauf

Jeweils eigenes Terminal, ROS/pixi-Env vorher sourcen
(`cd ~/ws_aic/src/aic && pixi shell`, ggf. `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
+ `ros2 run rmw_zenoh_cpp rmw_zenohd` in einem weiteren Terminal).

**Terminal 1 — Simulation mit Ground Truth + Task Board + Kabel:**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  spawn_task_board:=true \
  spawn_cable:=true \
  nic_card_mount_0_present:=true
```
`start_aic_engine` bewusst weglassen (default `false`) — der Engine würde
sonst selbst Lifecycle-Transitions/Goals übernehmen und den Model-Node bei
"Regelverstößen" killen.

**Terminal 2 — aic_model mit der Policy starten:**
```bash
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.SfpPlugInPhase1
```
Nach jeder Code-Änderung an der Policy **vorher** neu installieren (kein
Symlink-Install, `pixi-build-ros` kopiert die Datei wirklich):
```bash
pixi reinstall ros-kilted-aic-solution-policy
```

**Terminal 3 — Lifecycle aktivieren:**
```bash
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate
ros2 lifecycle get /aic_model   # zum Prüfen
```

**Terminal 4 — Roboter in Startpose bringen** (Tip-Standoff muss zu
`_insertion_offset_z` in der Policy passen, aktuell 5cm):
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p standoff_z:=0.05
```

**Terminal 5 — Goal senden:**
```bash
ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
  "{task: {id: 'test_1', cable_type: 'sfp_sc', cable_name: 'cable_0', plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', port_name: 'sfp_port_1', target_module_name: 'nic_card_mount_0', time_limit: 180}}" \
  --feedback
```
Die Task-Felder werden von der Policy inhaltlich ignoriert (nur `id` wird
geloggt), die Action braucht aber trotzdem eine gültige `Task`-Nachricht.

Fortschritt/Ergebnis siehst du live im Log von Terminal 2.

---

## 2. `position_over_port_debug.py` – alle Parameter

Alle Parameter über `--ros-args -p name:=wert`, mehrere kombinierbar.

| Parameter | Default | Bedeutung |
|---|---|---|
| `standoff_z` | `0.03` | Abstand Steckerspitze → Port-Eingang in Z (base_link/world), **muss zu `_insertion_offset_z`** in der Policy passen |
| `port_entrance_frame` | `task_board/nic_card_mount_0/sfp_port_1_link_entrance` | TF-Frame des Ziel-Ports |
| `cable_tip_frame` | `cable_0/sfp_tip_link` | TF-Frame der Steckerspitze |
| `tcp_frame` | `gripper/tcp` | TF-Frame des TCP |
| `base_frame` | `base_link` | Referenzframe für alle Berechnungen/Kommandos |
| `n_steps` | `100` | Interpolationsschritte für die Bewegung dorthin |
| `controller_namespace` | `aic_controller` | Namespace für `pose_commands`/`change_target_mode` |
| `simple_mode` | `false` | Debug: keine Orientierungs-/Tip-Offset-Rechnung, nur Translation, aktuelle TCP-Orientierung bleibt erhalten (zum isolierten Testen der Position) |
| `orientation_correction_rpy_deg` | `[0,0,0]` | Zusatzrotation (Grad, xyz) auf die Port-Orientierung, falls deren Z-Achse mal nicht wie erwartet zeigt |
| `xy_offset_min_m` / `xy_offset_max_m` | `0.0` / `0.0` | Simuliert Perception-Ungenauigkeit: zufälliger XY-Offset (unabhängig für X und Y aus `[min,max]` gesampelt) wird auf die Zielpose addiert und in der Konsole geloggt |
| `pose_backup_file` | `/tmp/position_over_port_debug_last_pose.json` | Wohin die aktuelle Pose vor jeder Bewegung gesichert wird |
| `reset` | `false` | Statt neu zu berechnen: zur zuletzt gesicherten Pose zurückfahren (Undo) |

### Beispiele

Normal positionieren (volle Pose inkl. Orientierung), 3cm Standoff:
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p standoff_z:=0.03
```

Nur Translation testen (Orientierung unverändert), 5cm über dem Port:
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p simple_mode:=true -p standoff_z:=0.05
```

Mit simulierter Perception-Ungenauigkeit (±1cm in X/Y):
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p standoff_z:=0.025 -p xy_offset_min_m:=-0.01 -p xy_offset_max_m:=0.01
```

Zurück zur Pose von vor dem letzten Lauf:
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p reset:=true
```

Falls die Steckerspitze falsch ausgerichtet aussieht:
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p orientation_correction_rpy_deg:="[180.0,0.0,0.0]"
```

---

## 3. Wichtigste Stellschrauben in `SfpPlugInPhase1.py`

Alles feste Python-Attribute im `__init__` (keine ROS-Parameter) — nach
Änderung `pixi reinstall ros-kilted-aic-solution-policy` nicht vergessen.

- `_insertion_offset_z` (5cm) — Zielpose = Startpose minus dieser Wert; muss zum `standoff_z` beim Positionieren passen
- `_contact_force_threshold_n` (10N) — Kraftschwelle für "Kontakt erkannt" beim Abstieg (tariert)
- `_max_descent_margin_m` (1cm) — Sicherheitsmarge über die Zielpose hinaus, bevor der Abstieg als gescheitert gilt
- `_entry_depth_threshold_m` (4mm) — wie weit TCP-z unter den **tatsächlich erkannten Kontaktpunkt** sinken muss, damit "Eintritt in den Port" gilt
- `_additional_insert_depth_m` (5cm) — Sicherheitsobergrenze fürs finale Einstecken (gestoppt wird früher, sobald z-Stillstand = vollständig eingesteckt erkannt wird)
- `_stall_window_steps` / `_stall_epsilon_m` / `_stall_grace_steps` — Stillstandserkennung (kein Fortschritt trotz tieferem Kommando = harter mechanischer Anschlag), genutzt sowohl bei der Kontakterkennung als auch beim finalen Einstecken
- `_cfg['descent_stiffness'/'descent_damping']` — Steifigkeit für die freie Abstiegsphase (steifer, um Kabelwiderstand zu überwinden)
- `_cfg['spiral_stiffness'/'spiral_damping'/'spiral_steps'/'spiral_max_radius'/'spiral_n_turns']` — unverändert unweich für Spiral-Search und finales Einstecken (Kraftbegrenzung am eigentlichen Port)
- `_debug_force_probe_only` — Debug-Schalter: wenn `True`, wird nur bis zur vollen `_insertion_offset_z`-Tiefe gefahren, jeder Schritt mit Fz geloggt, keine Kontakterkennung/Spiral/Insert (zum Kalibrieren von `_contact_force_threshold_n`)

---

## 4. Bekannte Stolperfallen

- **Kein Symlink-Install**: Änderungen an `.py`-Dateien wirken erst nach `pixi reinstall ros-kilted-aic-solution-policy`.
- **F/T-Sensor ist ungetart**: `observation.wrist_wrench` kommt roh von `/fts_broadcaster/wrench` (siehe `aic_adapter.cpp`), nicht vom controllerinternen Tare. Die Policy tariert deshalb selbst (Service-Aufruf best-effort + Software-Baseline über 20 Messwerte am Task-Start).
- **`task.time_limit`** wird von `aic_model` selbst nicht ausgewertet (nur `aic_engine` nutzt es als eigenen Abbruch-Timeout) — bei manuellem `ros2 action send_goal` zählt nur das interne Schrittbudget der Policy (`_descent_steps`, `_final_insert_steps`, Spiral-`spiral_steps`).
- **`standoff_z`** beim Debug-Skript muss zu `_insertion_offset_z` in der Policy passen, sonst landet die angenommene Zielpose weit vom echten Port entfernt.
