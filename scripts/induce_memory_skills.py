import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higraphmem2.skills.memory_skill import evolve_skills, induce_skills, save_skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--data-file", default="data/locomo/data/locomo10.json")
    parser.add_argument("--induction-count", default=1, type=int, help="Number of induction samples to use. Use 0 or a negative value for all samples in --data-file.")
    parser.add_argument("--enable-skill-evolution", action="store_true", help="Stream over induction samples and add evolved open-domain skills.")
    parser.add_argument("--output", default="skills/memory_skills.json")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    samples = json.loads((source_root / args.data_file).read_text(encoding="utf-8"))
    induction_samples = samples if args.induction_count <= 0 else samples[: args.induction_count]
    skills = induce_skills(induction_samples)
    if args.enable_skill_evolution:
        skills = evolve_skills(induction_samples, skills)
    save_skills(
        args.output,
        skills,
        metadata={
            "source_data": str(source_root / args.data_file),
            "induction_count": args.induction_count,
            "induction_sample_ids": [sample["sample_id"] for sample in induction_samples],
            "format": "train_induced_memory_skills",
            "skill_evolution_enabled": args.enable_skill_evolution,
            "note": "Memory skills selected/evolved from induction split questions only. Do not re-induce on test split.",
        },
    )
    print(args.output)
    print("skills:", ", ".join(skill.name for skill in skills))


if __name__ == "__main__":
    main()
