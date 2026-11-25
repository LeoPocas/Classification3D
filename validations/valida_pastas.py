import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


MAX_EPOCHS = 300
EXPECTED_RUNS_PER_GROUP = 15
DATASETS = ("incor", "mms")
MODEL_GROUPS = ("concat", "dual", "early_channel")
STATUS_PATTERNS = {
    "full": re.compile(r"(?<![a-z0-9])full(?![a-z0-9])", re.IGNORECASE),
    "earlyStop": re.compile(r"(?<![a-z0-9])earlystop(?![a-z0-9])", re.IGNORECASE),
}


@dataclass
class Finding:
    severity: str
    path: Path
    message: str


def _status_in_name(name: str) -> Optional[str]:
    matches = [
        status
        for status, pattern in STATUS_PATTERNS.items()
        if pattern.search(name)
    ]
    if len(matches) > 1:
        return "conflicting"
    return matches[0] if matches else None


def _status_for_file(path: Path, dataset_root: Path) -> Optional[str]:
    for parent in path.parents:
        if parent == dataset_root.parent:
            break
        status = _status_in_name(parent.name)
        if status:
            return status
    return _status_in_name(path.stem)


def _is_epoch(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_file(
    path: Path, dataset_root: Path, max_epochs: int
) -> list[Finding]:
    findings = []
    status = _status_for_file(path, dataset_root)
    folder_status = _status_in_name(path.parent.name)
    file_status = _status_in_name(path.stem)

    if folder_status == "conflicting" or file_status == "conflicting":
        findings.append(
            Finding("ERRO", path, "nome contém os rótulos full e earlyStop.")
        )
    elif folder_status and file_status and folder_status != file_status:
        findings.append(
            Finding(
                "ERRO",
                path,
                f"rótulo da pasta ({folder_status}) diverge do nome do arquivo "
                f"({file_status}).",
            )
        )

    try:
        with path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except (OSError, json.JSONDecodeError) as error:
        findings.append(
            Finding("ERRO", path, f"não foi possível ler o JSON: {error}")
        )
        return findings

    if not isinstance(data, dict):
        findings.append(Finding("ERRO", path, "a raiz do JSON deveria ser um objeto."))
        return findings

    config = data.get("config")
    results = data.get("results")

    if not isinstance(config, dict):
        findings.append(Finding("ERRO", path, "campo config ausente ou inválido."))
        config = {}
    if not isinstance(results, dict):
        findings.append(Finding("ERRO", path, "campo results ausente ou inválido."))
        results = {}

    configured_epochs = config.get("epochs")
    if configured_epochs is not None and (
        not _is_epoch(configured_epochs) or configured_epochs <= 0
    ):
        findings.append(
            Finding("ERRO", path, f"config.epochs inválido: {configured_epochs!r}.")
        )
        configured_epochs = None

    has_last_epoch = "last_epoch_executed" in results
    last_epoch = results.get("last_epoch_executed")
    effective_status = status
    if not has_last_epoch:
        # effective_status = "full"
        if status == "earlyStop":
            findings.append(
                Finding(
                    "AVISO",
                    path,
                    "pasta marcada earlyStop, mas last_epoch_executed ausente ",
                )
            )
    elif not _is_epoch(last_epoch) or last_epoch <= 0:
        findings.append(
            Finding(
                "ERRO",
                path,
                f"results.last_epoch_executed inválido: {last_epoch!r}.",
            )
        )
        last_epoch = None
    else:
        if last_epoch > max_epochs:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"last_epoch_executed={last_epoch} ultrapassa o máximo "
                    f"{max_epochs}.",
                )
            )
        if configured_epochs is not None and last_epoch > configured_epochs:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"last_epoch_executed={last_epoch} ultrapassa "
                    f"config.epochs={configured_epochs}.",
                )
            )

    if effective_status == "full":
        if last_epoch is not None and last_epoch != max_epochs:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"pasta marcada full, mas last_epoch_executed={last_epoch} "
                    f"(esperado {max_epochs}).",
                )
            )
        if configured_epochs is not None and configured_epochs != max_epochs:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"pasta marcada full, mas config.epochs={configured_epochs} "
                    f"(esperado {max_epochs}).",
                )
            )
    elif effective_status == "earlyStop" and last_epoch is not None:
        if last_epoch >= max_epochs:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"pasta marcada earlyStop, mas last_epoch_executed={last_epoch}; "
                    f"atingiu o máximo {max_epochs}.",
                )
            )

    early_stopping = config.get("early_stopping")
    if early_stopping is not None:
        if not isinstance(early_stopping, bool):
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"config.early_stopping deveria ser booleano, não "
                    f"{early_stopping!r}.",
                )
            )
        elif effective_status == "full" and early_stopping:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    "pasta marcada full, mas config.early_stopping=true.",
                )
            )
        elif effective_status == "earlyStop" and not early_stopping:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    "pasta marcada earlyStop, mas config.early_stopping=false.",
                )
            )

    best_epoch = results.get("best_epoch")
    if best_epoch is not None:
        if not _is_epoch(best_epoch) or best_epoch < 0:
            findings.append(
                Finding("ERRO", path, f"results.best_epoch inválido: {best_epoch!r}.")
            )
        elif last_epoch is not None and best_epoch > last_epoch:
            findings.append(
                Finding(
                    "ERRO",
                    path,
                    f"best_epoch={best_epoch} é posterior a "
                    f"last_epoch_executed={last_epoch}.",
                )
            )

    return findings


def validate(
    root: Path, max_epochs: int
) -> tuple[int, list[Finding], Counter[str]]:
    files = []
    findings = []
    files_by_group = Counter()

    dataset_roots = (
        [root]
        if root.name in DATASETS
        else [root / dataset for dataset in DATASETS]
    )
    for dataset_root in dataset_roots:
        dataset = dataset_root.name
        if not dataset_root.is_dir():
            findings.append(
                Finding(
                    "ERRO",
                    dataset_root,
                    "pasta do conjunto de dados não encontrada.",
                )
            )
            continue

        dataset_files = sorted(dataset_root.rglob("*.json"))
        files.extend(dataset_files)

        for model_group in MODEL_GROUPS:
            model_root = dataset_root / model_group
            if not model_root.is_dir():
                continue
            for execution_group in sorted(model_root.iterdir()):
                if not execution_group.is_dir():
                    continue
                run_count = sum(1 for _ in execution_group.glob("*.json"))
                if run_count != EXPECTED_RUNS_PER_GROUP:
                    findings.append(
                        Finding(
                            "ERRO",
                            execution_group,
                            f"grupo contém {run_count} execuções JSON; "
                            f"esperado {EXPECTED_RUNS_PER_GROUP}.",
                        )
                    )

        for path in dataset_files:
            relative_parts = path.relative_to(dataset_root).parts
            subgroup = relative_parts[0] if len(relative_parts) > 1 else "(raiz)"
            files_by_group[f"{dataset}/{subgroup}"] += 1
            findings.extend(_validate_file(path, dataset_root, max_epochs))

    return len(files), findings, files_by_group


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Confere rótulos full/earlyStop e metadados das execuções JSON "
            "em tempos/incor e tempos/mms."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tempos",
        help=(
            "pasta tempos que contém incor/ e mms/, ou diretamente "
            "tempos/incor ou tempos/mms."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"número máximo de épocas (padrão: {MAX_EPOCHS}).",
    )
    args = parser.parse_args()

    if args.max_epochs <= 0:
        parser.error("--max-epochs deve ser maior que zero.")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"diretório não encontrado: {root}")

    file_count, findings, files_by_group = validate(root, args.max_epochs)
    counts = Counter(finding.severity for finding in findings)

    print("JSON encontrados por dataset/subgrupo:")
    for group, count in sorted(files_by_group.items()):
        print(f"  {group}: {count}")

    for finding in findings:
        print(f"{finding.severity}: {finding.path}: {finding.message}")

    print(
        f"\nArquivos JSON verificados: {file_count} | "
        f"Erros: {counts['ERRO']} | Avisos: {counts['AVISO']} "
        f"| Execuções esperadas por pasta: {EXPECTED_RUNS_PER_GROUP}"
    )
    return 1 if counts["ERRO"] else 0


if __name__ == "__main__":
    raise SystemExit(main())