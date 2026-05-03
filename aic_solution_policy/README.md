# Plug_In Documentation

## Status:

- Force and Torque Wrench: /fts_broadcaster/wrench
- TF Frames for Port: task_board/nic_card_mount_0/sfp_port_0_link_entrance
- TF Frames for Plug Tip: cable_0/sfp_tip_link 
- TF Frames for TCP: gripper/tcp

To send Target Pose we need it from base_link or TCP

Bugs to fix next:
- Orientation alignment works but position is off
- Think again how to transorm from tcp to plug tip -> base link and back to entrence...


## Getting startet

### Testing single PlugIn
1. Terminal with (start simulation):

```bash
/entrypoint.sh   ground_truth:=true   start_aic_engine:=false   spawn_task_board:=true   nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.08 spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true

/entrypoint.sh   spawn_task_board:=true   sc_port_0_present:=true   sc_mount_rail_0_present:=true   spawn_cable:=true   cable_type:=sfp_sc_cable_reversed   attach_cable_to_gripper:=true   ground_truth:=true   start_aic_engine:=false
```

2. Terminal with (plugin package):
```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.PlugIn
```

3. Terminal with (plugin package):
```bash
cd ~/ws_aic/src/aic
pixi shell
ros2 lifecycle get /aic_model
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable 
    "{task: {
        id: 'cable_1', 
        cable_type: 'sfp', 
        cable_name: 'sfp_cable', 
        plug_type: 'sfp', 
        plug_name: 'sfp_plug', 
        port_type: 'sfp', 
        port_name: 'sfp_port_0', 
        target_module_name: 'nic_card_0', 
        time_limit: 60
            }
    }"

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
"{task: {
  id: 'cable_1',
  cable_type: 'sc',
  cable_name: 'sc_cable',
  plug_type: 'sc',
  plug_name: 'sc_plug',
  port_type: 'sc',
  port_name: 'sc_port_0',
  target_module_name: 'sc_mount_rail_0',
  time_limit: 60
}}"
```



### Testing Evaluation

1. Terminal 
```bash
cd ws_aic/src/aic 
pixi shell
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.PlugIn
```

2. Terminal
```bash
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```


## Developement Guide
To debug new package versionens 2. Terminal:
```bash
exit
pixi reinstall ros-kilted-aic-solution-policy
```
than again all steps from Terminal 3.


## Submission (Abgabe) – Schritt für Schritt

Diese Schritte bauen dein Submission-Image lokal, verifizieren es im lokalen Eval-Stack und laden es anschließend in die Challenge-Registry (ECR) hoch, damit du es im Submission-Portal einreichen kannst.

### Voraussetzungen
- Docker + Docker Compose Plugin
- AWS CLI installiert
- ECR Zugangsdaten + Team-Repository-URI aus der Onboarding-Mail
- Wichtig: Repository-Namen müssen **lowercase** sein (z.B. `diemanipulatoren`, nicht `DieManipulatoren`).
- Wichtig: ECR Image-Tags sind **immutable** → für jede neue Abgabe Tag erhöhen (`v1`, `v2`, …).

### 1) Prüfen, dass die richtige Policy im Image gestartet wird
- In `docker/my_policy/Dockerfile` ist die Policy über `CMD` gesetzt (z.B. `policy:=aic_solution_policy.VisionBasedSFPPlugIn`).
- In `docker/docker-compose.yaml` muss der `model`-Service auf dieses Dockerfile zeigen.
- Unser lokales Image heißt in der Regel `my-solution:v1` (nicht `localhost/my-solution:v1`).

### 2) Image lokal bauen
Aus dem AIC-Root:

```bash
cd ~/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml build model
```

Optionaler Check:

```bash
docker image ls | grep -E 'my-solution|solution'
```

### 3) Lokal verifizieren (Pflicht vor dem Push)

```bash
cd ~/ws_aic/src/aic
docker compose -f docker/docker-compose.yaml up
```

Erwartung:
- Der Stack startet
- `aic_engine` entdeckt `aic_model` und kann die Lifecycle-Services abfragen
- Der Trial startet (keine sofortige "Model validation failed" direkt beim Start)

Beenden:
- `Ctrl+C`

### 4) AWS-Profil konfigurieren (einmalig)

```bash
aws configure --profile <team_name>
export AWS_PROFILE=<team_name>
```

Region: `us-east-1`

### 5) Bei ECR einloggen

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
```

### 6) Taggen + Push (Teamname lowercase)

Beispiel (ersetze `<team_name_lowercase>`):

```bash
docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name_lowercase>:v1
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name_lowercase>:v1
```

Wenn `:v1` schon existiert (ECR tags immutable), nimm `:v2`, `:v3`, ...

### 7) Submission im Portal registrieren

1. Vollständige Image-URI kopieren (z.B. `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/diemanipulatoren:v1`).
2. Submission-Portal → Challenge → `Submit`
3. Phase `Qualification` auswählen
4. URI im Feld `OCI Image` einfügen
5. `Submit`


## Dokumentation: Submission-Fix (ohne Änderungen an aic_model)

### Kontext / Problem

In der Submission-Umgebung wird sehr früh geprüft, ob der Lifecycle-Node `aic_model` erreichbar ist und auf `GetState` antwortet.
Wenn dein Policy-Modul beim Import schwere Bibliotheken lädt (z.B. `ultralytics`/`torch`) oder direkt Modelle initialisiert, kann der Import lange dauern.
Das kann dazu führen, dass `GetState`-Aufrufe in der Engine timeouten und die Submission als "Model validation failed" endet (teilweise ohne hilfreiche Logs im Portal).

### Fix in diesem Package

Datei: `aic_solution_policy/VisionBasedSFPPlugIn.py`

- Heavy Imports (`ultralytics`/`torch`) sind nicht mehr im Modul-Top-Level, sondern werden lazily beim ersten YOLO-Use importiert.
- Das YOLO-Modell wird nicht mehr im Konstruktor sofort geladen, sondern erst beim ersten `detect_ports()`.
- `YOLO_CONFIG_DIR` wird auf `/tmp` gesetzt, damit Ultralytics seine Settings auch in restriktiven Container-Umgebungen schreiben kann.

Damit bleibt der Import/Configure-Teil schnell, und `aic_model` kann frühzeitig auf Lifecycle-Requests reagieren.

