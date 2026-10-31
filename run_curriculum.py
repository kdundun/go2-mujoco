"""
    python run_curriculum.py curriculum/c_flat_bridge_beams.yaml
    
Each stage starts from the final checkpoint of the previous stage.
Results:   logs/<env>-<timestamp>-<suffix>/  (suffix: set per stage, or <name>_<k>_<terrain>) 
Terminal:  run_logs/<name>_<k>_<terrain>.log
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

TRAIN = "learning/train_jax_ppo.py"

RESERVED = {"env", "steps", "overrides", "suffix"}


def as_number(v):
  """PyYAML reads e.g. 1e-4 as a string; turn numeric strings into numbers."""
  if isinstance(v, str):
    try:
      f = float(v.replace("_", ""))
      return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
      return v
  return v


def short_name(env):
  s = env.replace("Go2Goal", "").replace("Terrain", "")
  return s.lower()


def latest_run(suffix):
  runs = glob.glob(f"logs/*-{suffix}")
  if not runs:
    return None
  return max(runs, key=os.path.getmtime)


def build_cmd(cfg, k, stage, prev_ckpt):
  args = {**cfg.get("common", {}), **{a: b for a, b in stage.items() if a not in RESERVED}}
  overrides = dict(cfg.get("common_overrides", {}) or {})
  overrides.update(stage.get("overrides", {}) or {})
  camera = cfg.get("camera")
  if camera is not None:
    overrides["vision_config.enabled_geom_groups"] = list(camera)

  # Run name: set "suffix" in the stage to choose it, otherwise <name>_<k>_<terrain>
  suffix = stage.get("suffix") or f"{cfg['name']}_{k}_{short_name(stage['env'])}"
  cmd = [sys.executable, "-u", TRAIN,
         f"--env_name={stage['env']}",
         f"--num_timesteps={int(as_number(stage['steps']))}",
         f"--suffix={suffix}"]
  if camera is not None:
    cmd.append("--vision")
  for key, val in args.items():
    val = as_number(val)
    if isinstance(val, bool):
      cmd.append(f"--{key}" if val else f"--no{key}")
    else:
      cmd.append(f"--{key}={val}")
  if overrides:
    cmd.append("--playground_config_overrides=" + json.dumps(overrides))
  if prev_ckpt:
    cmd.append(f"--load_checkpoint_path={prev_ckpt}")
  return suffix, cmd


def run(cmd, logfile):
  with open(logfile, "w") as f:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
      sys.stdout.write(line)
      f.write(line)
    return p.wait()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("config")
  ap.add_argument("--dry_run", action="store_true", help="only print the commands")
  ap.add_argument("--from_stage", type=int, default=1,
                  help="start at this stage (1-based); earlier stages are taken from logs/")
  opt = ap.parse_args()

  cfg = yaml.safe_load(open(opt.config))
  stages = cfg["stages"]
  stamp = time.strftime("%Y%m%d-%H%M%S")
  os.makedirs("run_logs", exist_ok=True)
  summary = []

  prev_ckpt = cfg.get("init_checkpoint") 
  for k, stage in enumerate(stages, start=1):
    suffix, cmd = build_cmd(cfg, k, stage, prev_ckpt)

    if k < opt.from_stage:
      run_dir = latest_run(suffix)
      if run_dir is None and not opt.dry_run:
        sys.exit(f"--from_stage {opt.from_stage}: no run found for '{suffix}' in logs/")
      run_dir = run_dir or f"logs/<{suffix}>"
      print(f"=== stage {k} ({stage['env']}): reuse {run_dir}")
    else:
      print(f"\n=== stage {k}/{len(stages)}: {stage['env']}, {int(as_number(stage['steps'])):,} steps")
      print(" \\\n    ".join(cmd))
      if opt.dry_run:
        run_dir = f"logs/<{suffix}>"
      else:
        t0 = time.time()
        code = run(cmd, f"run_logs/{suffix}.log")
        if code != 0:
          sys.exit(f"stage {k} failed (exit {code}); see run_logs/{suffix}.log")
        run_dir = latest_run(suffix)
        shutil.copy(opt.config, os.path.join(run_dir, "curriculum.yaml"))
        with open(os.path.join(run_dir, "command.txt"), "w") as f:
          f.write(" ".join(cmd) + "\n")
        print(f"=== stage {k} done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")

    summary.append(f"stage {k}  {stage['env']:20s}  {run_dir}")
    prev_ckpt = f"{run_dir}/checkpoints"

  if not opt.dry_run:
    path = f"logs/curriculum_{cfg['name']}_{stamp}.txt"
    with open(path, "w") as f:
      f.write(f"config: {opt.config}\n" + "\n".join(summary) + "\n")
    print(f"\nsummary written to {path}")
  print("\n" + "\n".join(summary))


if __name__ == "__main__":
  main()
