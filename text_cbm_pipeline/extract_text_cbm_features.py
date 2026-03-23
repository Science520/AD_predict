#!/usr/bin/env python3
"""Build zero-shot concept bottleneck features for the Dementia Blog Corpus."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm.auto import tqdm

LOGGER = logging.getLogger("text_cbm_pipeline")

DEFAULT_XML_PATH = Path("~/AD_predict/blog_corpus-master/blog_corpus.xml").expanduser()
DEFAULT_FILTER_DIR = Path("~/AD_predict/blog_corpus-master/filters").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/AD_predict/text_cbm_pipeline").expanduser()
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "cbm_features.csv"

CONCEPT_SPECS: List[Tuple[str, str]] = [
    # Emotional and behavioral themes.
    ("concept_emotional_frustration_anxiety", "emotional frustration or anxiety"),
    ("concept_health_issues_doctors", "talking about health issues or doctors"),
    ("concept_nostalgia_distant_past", "nostalgia or talking about the distant past"),
    # Linguistic simplification and cognitive markers.
    ("concept_simple_vocabulary", "very simple and basic vocabulary"),
    ("concept_repetitive_language", "repetitive words or sentences"),
    ("concept_grammatical_errors_confused_syntax", "grammatical errors or confused syntax"),
    ("concept_logical_discontinuity", "lack of logical flow or sudden topic changes"),
    ("concept_short_fragmented_sentences", "short and fragmented sentences"),
]

OUTPUT_COLUMNS = [
    "blog_name",
    "date",
    "clean_text",
    "concept_emotional_frustration_anxiety",
    "concept_health_issues_doctors",
    "concept_nostalgia_distant_past",
    "concept_simple_vocabulary",
    "concept_repetitive_language",
    "concept_grammatical_errors_confused_syntax",
    "concept_logical_discontinuity",
    "concept_short_fragmented_sentences",
]


@dataclass(frozen=True)
class ParsedPost:
    blog_name: str
    date: str
    clean_text: str


@dataclass(frozen=True)
class ParseStats:
    total_posts: int
    excluded_posts: int
    kept_posts: int
    parser_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract filtered blog posts and zero-shot concept bottleneck features."
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--filters-dir", type=Path, default=DEFAULT_FILTER_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--model-name",
        default="facebook/bart-large-mnli",
        help="HuggingFace zero-shot model name.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Transformers pipeline device index. Use -1 for CPU.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Truncate each post to this many characters before zero-shot inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of posts scored together in each zero-shot batch.",
    )
    parser.add_argument(
        "--limit-posts",
        type=int,
        default=None,
        help="Optional debug cap on retained posts after filtering.",
    )
    parser.add_argument(
        "--hypothesis-template",
        default="The content or writing style of this blog post shows {}.",
        help="Hypothesis template for zero-shot classification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and filter the corpus, then exit before zero-shot inference.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def strip_wrapping_quotes(value: str) -> str:
    return value.strip().strip("\"'").strip()


def normalize_blog_identifier(value: str) -> str:
    cleaned = strip_wrapping_quotes(value)
    if cleaned.startswith("#"):
        cleaned = strip_wrapping_quotes(cleaned[1:])
    cleaned = cleaned.rstrip("/")
    cleaned = cleaned.replace(".blogspot.ca", ".blogspot.com")
    return cleaned.lower()


def normalize_date(value: str) -> str:
    cleaned = strip_wrapping_quotes(value)
    if not cleaned:
        return ""
    try:
        parsed = datetime.strptime(cleaned, "%A, %B %d, %Y")
        return parsed.strftime("%A, %B %d, %Y")
    except ValueError:
        return cleaned


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_filter_blocklist(filters_dir: Path) -> Set[Tuple[str, str]]:
    if not filters_dir.exists():
        raise FileNotFoundError(f"Filters directory does not exist: {filters_dir}")

    filter_files = sorted(filters_dir.glob("*.txt"))
    if not filter_files:
        raise FileNotFoundError(f"No filter .txt files found in: {filters_dir}")

    blocklist: Set[Tuple[str, str]] = set()
    for filter_path in filter_files:
        current_blog = None
        for line_number, raw_line in enumerate(
            filter_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = strip_wrapping_quotes(raw_line)
            if not line:
                continue
            if line.startswith("#"):
                current_blog = normalize_blog_identifier(line)
                continue
            if current_blog is None:
                LOGGER.warning(
                    "Skipping orphan date line in %s:%s -> %r",
                    filter_path,
                    line_number,
                    raw_line,
                )
                continue
            blocklist.add((current_blog, normalize_date(line)))
    LOGGER.info("Loaded %d exclusion tuples from %d filter files.", len(blocklist), len(filter_files))
    return blocklist


def choose_bs4_parser() -> str:
    if importlib.util.find_spec("lxml") is not None:
        return "xml"
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    LOGGER.warning(
        "lxml is not installed; falling back to BeautifulSoup's html.parser. "
        "Install lxml for stricter XML parsing."
    )
    return "html.parser"


def reconstruct_post_text(post_node) -> str:
    sentence_nodes = sorted(
        post_node.find_all("sentence", recursive=False),
        key=lambda node: int(node.get("id", 10**9))
        if str(node.get("id", "")).isdigit()
        else 10**9,
    )
    sentences = [
        clean_whitespace(sentence.get_text(" ", strip=True))
        for sentence in sentence_nodes
    ]
    return clean_whitespace(" ".join(part for part in sentences if part))


def parse_xml_posts(
    xml_path: Path,
    blocklist: Set[Tuple[str, str]],
    limit_posts: int | None = None,
) -> Tuple[List[ParsedPost], ParseStats]:
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file does not exist: {xml_path}")

    xml_text = xml_path.read_text(encoding="utf-8")
    total_posts = xml_text.count("<post ")
    parser_name = choose_bs4_parser()
    soup = BeautifulSoup(xml_text, parser_name)

    retained_posts: List[ParsedPost] = []
    excluded_posts = 0
    progress = tqdm(total=total_posts, desc="Parsing XML posts", unit="post")

    try:
        for blog_node in soup.find_all("blog"):
            raw_blog_name = clean_whitespace(blog_node.get("name", ""))
            blog_key = normalize_blog_identifier(raw_blog_name)

            for post_node in blog_node.find_all("post", recursive=False):
                progress.update(1)

                raw_date = strip_wrapping_quotes(post_node.get("date", ""))
                date_key = normalize_date(raw_date)
                if (blog_key, date_key) in blocklist:
                    excluded_posts += 1
                    continue

                clean_text = reconstruct_post_text(post_node)
                if not clean_text:
                    continue

                retained_posts.append(
                    ParsedPost(
                        blog_name=raw_blog_name,
                        date=date_key or raw_date,
                        clean_text=clean_text,
                    )
                )
                if limit_posts is not None and len(retained_posts) >= limit_posts:
                    LOGGER.info("Reached debug limit of %d retained posts.", limit_posts)
                    break
            if limit_posts is not None and len(retained_posts) >= limit_posts:
                break
    finally:
        progress.close()

    stats = ParseStats(
        total_posts=total_posts,
        excluded_posts=excluded_posts,
        kept_posts=len(retained_posts),
        parser_name=parser_name,
    )
    LOGGER.info(
        "Parsed %d posts, excluded %d, retained %d using parser=%s.",
        stats.total_posts,
        stats.excluded_posts,
        stats.kept_posts,
        stats.parser_name,
    )
    return retained_posts, stats


def chunked(items: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def truncate_for_inference(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return clean_whitespace(text[:max_chars])


def build_zero_shot_classifier(model_name: str, device: int):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for zero-shot inference. Install torch and transformers."
        ) from exc

    if device >= 0 and not torch.cuda.is_available():
        LOGGER.warning("CUDA device=%d requested but no GPU is available. Falling back to CPU.", device)
        device = -1

    # If you hit CUDA OOM with device=0, change it to device=-1 to run on CPU.
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    return classifier, device


def score_posts_zero_shot(
    posts: Sequence[ParsedPost],
    model_name: str,
    device: int,
    batch_size: int,
    max_chars: int,
    hypothesis_template: str,
) -> List[Dict[str, float]]:
    if not posts:
        return []

    classifier, resolved_device = build_zero_shot_classifier(model_name=model_name, device=device)
    LOGGER.info(
        "Running zero-shot inference on %d posts with model=%s device=%d.",
        len(posts),
        model_name,
        resolved_device,
    )

    concept_labels = [concept for _, concept in CONCEPT_SPECS]
    truncated_texts = [truncate_for_inference(post.clean_text, max_chars) for post in posts]
    all_scores: List[Dict[str, float]] = []

    progress = tqdm(total=len(posts), desc="Zero-shot concept scoring", unit="post")
    try:
        for batch_texts in chunked(truncated_texts, batch_size=batch_size):
            outputs = classifier(
                list(batch_texts),
                candidate_labels=concept_labels,
                hypothesis_template=hypothesis_template,
                multi_label=True,
                truncation=True,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]

            for output in outputs:
                score_lookup = {
                    label: float(score)
                    for label, score in zip(output["labels"], output["scores"])
                }
                all_scores.append(
                    {
                        column_name: score_lookup.get(concept_name, 0.0)
                        for column_name, concept_name in CONCEPT_SPECS
                    }
                )
            progress.update(len(batch_texts))
    finally:
        progress.close()

    return all_scores


def build_output_dataframe(
    posts: Sequence[ParsedPost],
    concept_scores: Sequence[Dict[str, float]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for index, post in enumerate(posts):
        row: Dict[str, object] = {
            "blog_name": post.blog_name,
            "date": post.date,
            "clean_text": post.clean_text,
        }
        if concept_scores:
            row.update(concept_scores[index])
        else:
            row.update({column_name: None for column_name, _ in CONCEPT_SPECS})
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be a positive integer.")
    if args.limit_posts is not None and args.limit_posts <= 0:
        raise ValueError("--limit-posts must be a positive integer when provided.")

    args.xml_path = args.xml_path.expanduser().resolve()
    args.filters_dir = args.filters_dir.expanduser().resolve()
    args.output_csv = args.output_csv.expanduser().resolve()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    blocklist = parse_filter_blocklist(args.filters_dir)
    posts, stats = parse_xml_posts(
        xml_path=args.xml_path,
        blocklist=blocklist,
        limit_posts=args.limit_posts,
    )

    if args.dry_run:
        LOGGER.info(
            "Dry run complete. Retained %d posts after filtering. No model inference was run.",
            stats.kept_posts,
        )
        return 0

    concept_scores = score_posts_zero_shot(
        posts=posts,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        hypothesis_template=args.hypothesis_template,
    )

    if len(concept_scores) != len(posts):
        raise RuntimeError(
            f"Score count mismatch: expected {len(posts)} rows but received {len(concept_scores)}."
        )

    output_df = build_output_dataframe(posts=posts, concept_scores=concept_scores)
    output_df.to_csv(args.output_csv, index=False)
    LOGGER.info("Saved %d rows to %s", len(output_df), args.output_csv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
