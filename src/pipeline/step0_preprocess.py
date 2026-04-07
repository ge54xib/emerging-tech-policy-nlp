"""Step 0: PDF preprocessing via Adobe PDF Extract API.

This step converts each PDF policy document into:
- structured JSON (`data/output/step0_preprocessing/json/*.json`)
- plain text (`data/output/step0_preprocessing/text/*.txt`)
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from src import config


def _require_credentials() -> None:
    if config.ADOBE_CLIENT_ID and config.ADOBE_CLIENT_SECRET:
        return
    raise RuntimeError(
        "Missing Adobe credentials. Set ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET."
    )


def _load_structured_json(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open("structuredData.json") as handle:
            return json.load(handle)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_index(segment: str) -> str:
    return re.sub(r"\[\d+\]$", "", segment)


def _base_tag(path: str) -> str:
    segments = [seg for seg in (path or "").split("/") if seg]
    if not segments:
        return ""
    return _strip_index(segments[-1])


def _path_tags(path: str) -> set[str]:
    tags: set[str] = set()
    for segment in (path or "").split("/"):
        if not segment:
            continue
        tag = _strip_index(segment).strip()
        if tag:
            tags.add(tag.upper())
    return tags


def _group_path(path: str) -> str:
    """Group sub-spans back to a parent structural block."""
    if not path:
        return path
    grouped = re.sub(r"/(Sub|Span|Reference|Lbl|LBody)(\[\d+\])?$", "", path)
    return grouped


def _block_type(path: str) -> str:
    if _is_toc_path(path):
        return "toc_item"
    if _is_footnote_path(path):
        return "footnote"
    tag = _base_tag(path)
    if tag.startswith("H") and tag[1:].isdigit():
        return f"heading_{tag[1:]}"
    if tag == "P":
        return "paragraph"
    if tag == "LI":
        return "list_item"
    if tag in {"TH", "TD", "Table"}:
        return "table"
    if tag in {"Title"}:
        return "heading_1"
    return "other"


def _is_footnote_path(path: str) -> bool:
    if not path:
        return False
    return re.search(r"(^|/)(footnote|fn)(\[\d+\])?($|/)", path.lower()) is not None


def _is_toc_path(path: str) -> bool:
    if not path:
        return False
    return "/toc/toci" in path.lower()


def _extract_structured_blocks(payload: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []

    for element in payload.get("elements", []):
        raw_text = element.get("Text")
        if not raw_text:
            continue
        text = _normalize_text(raw_text)
        if not text:
            continue

        raw_path = element.get("Path", "")
        path = _group_path(raw_path)
        if path not in grouped:
            grouped[path] = {
                "path": path,
                "page": element.get("Page"),
                "type": _block_type(path),
                "fragments": [],
            }
            order.append(path)
        grouped[path]["fragments"].append(text)

    blocks: list[dict] = []
    for idx, path in enumerate(order):
        info = grouped[path]
        merged_text = _normalize_text(" ".join(info["fragments"]))
        if not merged_text:
            continue
        block = {
            "block_id": idx,
            "path": info["path"],
            "page": info["page"],
            "type": info["type"],
            "text": merged_text,
        }
        blocks.append(block)
    return blocks


def _country_code_from_stem(stem: str) -> str:
    raw = (stem or "").strip()
    if not raw:
        return ""
    return raw.split("_", 1)[0].upper()


def _excluded_pages_for_stem(stem: str) -> set[int]:
    pages = set(config.PREPROCESS_EXCLUDE_PAGES)
    country = _country_code_from_stem(stem)
    if country:
        pages |= config.PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY.get(country, set())
    pages |= config.PREPROCESS_EXCLUDE_PAGES_BY_DOC.get(stem.lower(), set())
    return pages


def _filter_blocks(
    blocks: list[dict],
    *,
    excluded_pages: set[int] | None = None,
    excluded_tags: set[str] | None = None,
) -> list[dict]:
    """Apply path-tag based pruning before writing structured/text outputs."""
    filtered: list[dict] = []
    active_excluded_tags = (
        set(excluded_tags)
        if excluded_tags is not None
        else set(config.PREPROCESS_EXCLUDE_PATH_TAGS)
    )
    active_excluded_pages = (
        set(excluded_pages)
        if excluded_pages is not None
        else set(config.PREPROCESS_EXCLUDE_PAGES)
    )
    for block in blocks:
        page = block.get("page")
        page_num: int | None
        try:
            page_num = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_num = None
        if page_num is not None and page_num in active_excluded_pages:
            continue

        tags = _path_tags(str(block.get("path", "")))
        if tags & active_excluded_tags:
            continue
        filtered.append(block)

    for idx, block in enumerate(filtered):
        block["block_id"] = idx
    return filtered


def _blocks_to_text(blocks: list[dict]) -> str:
    """Render structured blocks as readable plain text."""
    lines: list[str] = []

    for block in blocks:
        block_type = block["type"]
        text = block["text"]
        if not text:
            continue

        if block_type.startswith("heading_"):
            level = block_type.split("_")[-1]
            try:
                level_num = max(1, min(int(level), 3))
            except ValueError:
                level_num = 1
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f'{"#" * level_num} {text}')
            lines.append("")
            continue

        if block_type == "list_item":
            lines.append(f"- {text}")
            continue

        if block_type == "table":
            lines.append(f"[TABLE] {text}")
            continue

        lines.append(text)

    # Compact repeated empty lines while preserving paragraph boundaries.
    compact: list[str] = []
    for line in lines:
        if line == "" and compact and compact[-1] == "":
            continue
        compact.append(line)
    return "\n".join(compact).strip() + "\n"


def _write_structure_outputs(stem: str, payload: dict) -> tuple[Path, Path, Path]:
    json_output = config.STEP0_JSON_DIR / f"{stem}.json"
    structured_output = config.STEP0_STRUCTURED_DIR / f"{stem}.json"
    text_output = config.STEP0_TEXT_DIR / f"{stem}.txt"

    blocks = _filter_blocks(
        _extract_structured_blocks(payload),
        excluded_pages=_excluded_pages_for_stem(stem),
    )
    rendered_text = _blocks_to_text(blocks)

    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    structured_output.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    text_output.write_text(rendered_text, encoding="utf-8")
    return json_output, structured_output, text_output


def _stream_to_bytes(payload) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if hasattr(payload, "read"):
        return payload.read()
    raise TypeError("Unsupported stream payload type returned by Adobe SDK.")


def _extract_with_new_sdk(pdf_path: Path, zip_output: Path) -> None:
    """Adobe PDF Services SDK v4+ flow."""
    from adobe.pdfservices.operation.auth.service_principal_credentials import (
        ServicePrincipalCredentials,
    )
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
        ExtractElementType,
    )
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
        ExtractPDFParams,
    )
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
        ExtractPDFResult,
    )

    with pdf_path.open("rb") as handle:
        source_bytes = handle.read()

    credentials = ServicePrincipalCredentials(
        client_id=config.ADOBE_CLIENT_ID,
        client_secret=config.ADOBE_CLIENT_SECRET,
    )
    pdf_services = PDFServices(credentials=credentials)
    input_asset = pdf_services.upload(
        input_stream=source_bytes,
        mime_type=PDFServicesMediaType.PDF,
    )
    extract_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES]
    )
    extract_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_params)

    polling_url = pdf_services.submit(extract_job)
    response = pdf_services.get_job_result(polling_url, ExtractPDFResult)
    resource_asset = response.get_result().get_resource()
    stream_asset = pdf_services.get_content(resource_asset)
    zip_output.write_bytes(_stream_to_bytes(stream_asset.get_input_stream()))


def _extract_with_legacy_sdk(pdf_path: Path, zip_output: Path) -> None:
    """Fallback for older Adobe SDK releases."""
    from adobe.pdfservices.operation.auth.credentials import Credentials
    from adobe.pdfservices.operation.execution_context import ExecutionContext
    from adobe.pdfservices.operation.io.file_ref import FileRef
    from adobe.pdfservices.operation.pdfops.extract_pdf_operation import (
        ExtractPDFOperation,
    )
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import (
        ExtractElementType,
    )
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import (
        ExtractPDFOptions,
    )

    credentials = (
        Credentials.service_principal_credentials_builder()
        .with_client_id(config.ADOBE_CLIENT_ID)
        .with_client_secret(config.ADOBE_CLIENT_SECRET)
        .build()
    )
    execution_context = ExecutionContext.create(credentials)
    operation = ExtractPDFOperation.create_new()
    operation.set_input(FileRef.create_from_local_file(str(pdf_path)))
    options = (
        ExtractPDFOptions.builder()
        .with_elements_to_extract([ExtractElementType.TEXT, ExtractElementType.TABLES])
        .build()
    )
    operation.set_options(options)
    operation.execute(execution_context).save_as(str(zip_output))


def _extract_single_pdf(pdf_path: Path) -> tuple[Path, Path]:
    """Run Adobe Extract on one PDF and return (json_path, text_path)."""
    _require_credentials()

    from adobe.pdfservices.operation.exception.exceptions import (
        SdkException,
        ServiceApiException,
        ServiceUsageException,
    )

    stem = pdf_path.stem
    zip_output = config.STEP0_JSON_DIR / f"{stem}.zip"

    try:
        try:
            _extract_with_new_sdk(pdf_path, zip_output)
        except ModuleNotFoundError:
            _extract_with_legacy_sdk(pdf_path, zip_output)
    except (SdkException, ServiceApiException, ServiceUsageException) as exc:
        raise RuntimeError(f"Adobe Extract failed for {pdf_path.name}: {exc}") from exc

    payload = _load_structured_json(zip_output)
    json_output, _structured_output, text_output = _write_structure_outputs(stem, payload)
    return json_output, text_output


def run() -> None:
    print(">>> STEP 0: Adobe PDF Extract preprocessing")
    pdf_paths = sorted(config.INPUT_PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {config.INPUT_PDF_DIR}")

    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        cached_json_path = config.STEP0_JSON_DIR / f"{stem}.json"

        if cached_json_path.exists():
            payload = json.loads(cached_json_path.read_text(encoding="utf-8"))
            json_path, structured_path, text_path = _write_structure_outputs(stem, payload)
            print(
                f"[OK] {pdf_path.name} -> {json_path.name}, {structured_path.name}, {text_path.name} (from existing JSON)"
            )
            continue

        json_path, text_path = _extract_single_pdf(pdf_path)
        print(f"[OK] {pdf_path.name} -> {json_path.name}, {text_path.name}")


if __name__ == "__main__":
    run()
