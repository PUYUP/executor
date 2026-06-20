"""Mapper that fuses GROBID TEI output with Docling visual assets.

GROBID produces structured text (title, authors, abstract, sections,
references) as TEI XML.  Docling produces visual assets (figure images,
table images, formula text) with page numbers and bounding boxes.

This module:
  1. Parses the GROBID TEI XML into a clean ``GrobidDocument``.
  2. Matches each Docling ``ExtractedAsset`` to the nearest TEI figure /
     table element using page number + bounding-box overlap (IoU).
  3. Returns a single ``FusedDocument`` that carries both the structured
     text and the enriched visual assets in one place.

Coordinate systems
------------------
GROBID TEI ``<figure>`` / ``<table>`` elements may carry ``<figDesc>``
coordinates encoded as ``coords="page,x1,y1,x2,y2"`` (PDF points, origin
top-left in GROBID).  Docling bboxes are also in PDF points but with
origin **bottom-left** (CoordOrigin.BOTTOMLEFT in Docling 2.x).

The Y axis is flipped before IoU is computed:
    y_topleft = page_height_pt - y_bottomleft

Default page height is A4 (841.89 pt).  Pass ``page_height_pt`` to
``DocumentMapper.map()`` if you know the actual page height.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from executor.docling_exec import AssetType, DoclingResult, ExtractedAsset

logger = logging.getLogger(__name__)

__all__ = [
    "FusedDocument",
    "FusedAsset",
    "GrobidDocument",
    "GrobidAuthor",
    "GrobidReference",
    "GrobidSection",
    "GrobidFigureRef",
    "DocumentMapper",
]

# TEI namespace used by GROBID
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


# ---------------------------------------------------------------------------
# GROBID parsed types
# ---------------------------------------------------------------------------


@dataclass
class GrobidAuthor:
    first_name: str | None = None
    last_name: str | None = None
    affiliation: str | None = None
    email: str | None = None

    @property
    def full_name(self) -> str:
        parts = filter(None, [self.first_name, self.last_name])
        return " ".join(parts)


@dataclass
class GrobidReference:
    ref_id: str | None = None
    title: str | None = None
    authors: list[GrobidAuthor] = field(default_factory=list)
    year: str | None = None
    journal: str | None = None
    volume: str | None = None
    pages: str | None = None
    doi: str | None = None
    raw_text: str | None = None


@dataclass
class GrobidFigureRef:
    """A <figure> or <table> element found in the TEI body."""

    fig_id: str | None = None
    label: str | None = None
    caption: str | None = None
    coords: tuple[int, float, float, float, float] | None = None
    # (page, x1, y1, x2, y2) — page is 1-based, top-left origin, PDF points


@dataclass
class GrobidSection:
    heading: str | None = None
    text: str = ""


@dataclass
class GrobidDocument:
    """Structured output parsed from a GROBID TEI XML string."""

    title: str | None = None
    abstract: str | None = None
    authors: list[GrobidAuthor] = field(default_factory=list)
    sections: list[GrobidSection] = field(default_factory=list)
    references: list[GrobidReference] = field(default_factory=list)
    figure_refs: list[GrobidFigureRef] = field(default_factory=list)
    raw_xml: str = ""


# ---------------------------------------------------------------------------
# Fused types
# ---------------------------------------------------------------------------


@dataclass
class FusedAsset:
    """A Docling visual asset enriched with data from GROBID TEI."""

    asset: ExtractedAsset
    grobid_ref: GrobidFigureRef | None = None
    match_confidence: float = 0.0

    @property
    def caption(self) -> str | None:
        """Best available caption: Docling first, GROBID fallback."""
        return self.asset.caption or (
            self.grobid_ref.caption if self.grobid_ref else None
        )

    @property
    def label(self) -> str | None:
        return self.grobid_ref.label if self.grobid_ref else None


@dataclass
class FusedDocument:
    """A single document combining GROBID text structure and Docling visuals."""

    grobid: GrobidDocument
    assets: list[FusedAsset] = field(default_factory=list)

    @property
    def figures(self) -> list[FusedAsset]:
        return [a for a in self.assets if a.asset.asset_type == AssetType.PICTURE]

    @property
    def tables(self) -> list[FusedAsset]:
        return [a for a in self.assets if a.asset.asset_type == AssetType.TABLE]

    @property
    def formulas(self) -> list[FusedAsset]:
        return [a for a in self.assets if a.asset.asset_type == AssetType.FORMULA]

    @property
    def matched_assets(self) -> list[FusedAsset]:
        return [a for a in self.assets if a.grobid_ref is not None]

    @property
    def unmatched_assets(self) -> list[FusedAsset]:
        return [a for a in self.assets if a.grobid_ref is None]


# ---------------------------------------------------------------------------
# TEI parser
# ---------------------------------------------------------------------------


class _TeiParser:
    """Parses a GROBID TEI XML string into a GrobidDocument."""

    def parse(self, xml: str) -> GrobidDocument:
        root = ET.fromstring(xml)
        return GrobidDocument(
            title=self._title(root),
            abstract=self._abstract(root),
            authors=self._authors(root),
            sections=self._sections(root),
            references=self._references(root),
            figure_refs=self._figure_refs(root),
            raw_xml=xml,
        )

    def _title(self, root: ET.Element) -> str | None:
        el = root.find(".//tei:titleStmt/tei:title", _NS)
        return self._text(el)

    def _abstract(self, root: ET.Element) -> str | None:
        el = root.find(".//tei:abstract", _NS)
        if el is None:
            return None
        return " ".join(el.itertext()).strip() or None

    def _authors(self, root: ET.Element) -> list[GrobidAuthor]:
        authors: list[GrobidAuthor] = []
        for person in root.findall(".//tei:sourceDesc//tei:author", _NS):
            fn = person.find("tei:persName/tei:forename", _NS)
            sn = person.find("tei:persName/tei:surname", _NS)
            aff = person.find(".//tei:affiliation/tei:orgName", _NS)
            em = person.find(".//tei:email", _NS)
            authors.append(GrobidAuthor(
                first_name=self._text(fn),
                last_name=self._text(sn),
                affiliation=self._text(aff),
                email=self._text(em),
            ))
        return authors

    def _sections(self, root: ET.Element) -> list[GrobidSection]:
        sections: list[GrobidSection] = []
        body = root.find(".//tei:body", _NS)
        if body is None:
            return sections
        for div in body.findall("tei:div", _NS):
            heading_el = div.find("tei:head", _NS)
            heading = self._text(heading_el)
            paras: list[str] = []
            for p in div.findall("tei:p", _NS):
                txt = " ".join(p.itertext()).strip()
                if txt:
                    paras.append(txt)
            sections.append(GrobidSection(
                heading=heading,
                text="\n\n".join(paras),
            ))
        return sections

    def _references(self, root: ET.Element) -> list[GrobidReference]:
        refs: list[GrobidReference] = []
        list_bibl = root.find(".//tei:listBibl", _NS)
        if list_bibl is None:
            return refs
        for bibl in list_bibl.findall("tei:biblStruct", _NS):
            ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id")
            title_el = (
                bibl.find(".//tei:title[@level='a']", _NS)
                or bibl.find(".//tei:title", _NS)
            )
            year_el = bibl.find(".//tei:date", _NS)
            journal_el = bibl.find(".//tei:title[@level='j']", _NS)
            volume_el = bibl.find(".//tei:biblScope[@unit='volume']", _NS)
            pages_el = bibl.find(".//tei:biblScope[@unit='page']", _NS)
            doi_el = bibl.find(".//tei:idno[@type='DOI']", _NS)
            authors: list[GrobidAuthor] = []
            for person in bibl.findall(".//tei:author", _NS):
                fn = person.find("tei:persName/tei:forename", _NS)
                sn = person.find("tei:persName/tei:surname", _NS)
                authors.append(GrobidAuthor(
                    first_name=self._text(fn),
                    last_name=self._text(sn),
                ))
            raw = " ".join(bibl.itertext()).strip()
            refs.append(GrobidReference(
                ref_id=ref_id,
                title=self._text(title_el),
                authors=authors,
                year=year_el.get("when") if year_el is not None else None,
                journal=self._text(journal_el),
                volume=self._text(volume_el),
                pages=self._text(pages_el),
                doi=self._text(doi_el),
                raw_text=raw or None,
            ))
        return refs

    def _figure_refs(self, root: ET.Element) -> list[GrobidFigureRef]:
        refs: list[GrobidFigureRef] = []
        for fig in root.findall(".//tei:figure", _NS):
            fig_id = fig.get("{http://www.w3.org/XML/1998/namespace}id")
            label_el = fig.find("tei:head", _NS)
            caption_el = fig.find("tei:figDesc", _NS)
            coords = self._parse_coords(fig.get("coords"))
            refs.append(GrobidFigureRef(
                fig_id=fig_id,
                label=self._text(label_el),
                caption=self._text(caption_el),
                coords=coords,
            ))
        return refs

    @staticmethod
    def _text(el: ET.Element | None) -> str | None:
        if el is None:
            return None
        return " ".join(el.itertext()).strip() or None

    @staticmethod
    def _parse_coords(
        coords_str: str | None,
    ) -> tuple[int, float, float, float, float] | None:
        """Parse GROBID ``coords="page,x1,y1,x2,y2"`` attribute.

        GROBID may emit multiple spans separated by semicolons; we take
        the bounding box of ALL spans so the full element is covered.
        """
        if not coords_str:
            return None
        spans = coords_str.split(";")
        pages: list[int] = []
        x1s: list[float] = []
        y1s: list[float] = []
        x2s: list[float] = []
        y2s: list[float] = []
        for span in spans:
            parts = span.strip().split(",")
            if len(parts) != 5:
                continue
            try:
                pages.append(int(parts[0]))
                x1s.append(float(parts[1]))
                y1s.append(float(parts[2]))
                x2s.append(float(parts[3]))
                y2s.append(float(parts[4]))
            except ValueError:
                continue
        if not pages:
            return None
        # Use the first page; union all bboxes
        return (pages[0], min(x1s), min(y1s), max(x2s), max(y2s))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) rectangles."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _caption_similarity(a: str | None, b: str | None) -> float:
    """Token-overlap Jaccard similarity (0.0–1.0), stopwords removed."""
    if not a or not b:
        return 0.0
    _STOPS = frozenset({
        "figure", "fig", "table", "tab",
        "the", "a", "an", "of", "in", "and", "is", "are", "for",
    })
    tok_a = set(re.findall(r"\w+", a.lower())) - _STOPS
    tok_b = set(re.findall(r"\w+", b.lower())) - _STOPS
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def _ordinal_numbers(text: str) -> list[str]:
    """Extract all digit sequences from a label/caption string."""
    return re.findall(r"\d+", text)


# ---------------------------------------------------------------------------
# Public mapper
# ---------------------------------------------------------------------------


# Standard page heights in PDF points (1 pt = 1/72 inch)
PAGE_HEIGHT_A4 = 841.89
PAGE_HEIGHT_LETTER = 792.0


class DocumentMapper:
    """Fuses a GROBID TEI string and a DoclingResult into a FusedDocument.

    Matching strategy (scores accumulate; highest-scoring pair wins):

    Priority  Signal                       Score range   Condition
    --------  ------                       -----------   ---------
    1 (hard)  Bbox IoU ≥ iou_threshold     0.60–1.00     same page, both have bbox
    2         Bbox IoU 0 < x < threshold   0.25–0.59     same page, partial overlap
    3         Same page, no bbox           0.15          one/both sides lack bbox
    4         Label ordinal match          +0.55–0.70    "Figure 2" ↔ caption "2"
    5         Caption token similarity     0.35–0.90     Jaccard ≥ caption_threshold

    Assignment is globally greedy: scores are computed for all (asset, ref)
    pairs, then matched highest-score-first so early assets don't starve
    later ones.

    Y-axis normalisation
    --------------------
    Docling bbox origin is bottom-left; GROBID is top-left.  We flip
    Docling Y before computing IoU::

        y_tl = page_height_pt - y_bl

    Pass the actual page height via ``map(..., page_height_pt=...)`` for
    precision.  The default (A4, 841.89 pt) is close enough for most papers.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        caption_threshold: float = 0.1,
        page_height_pt: float = PAGE_HEIGHT_A4,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._caption_threshold = caption_threshold
        self._page_height = page_height_pt
        self._parser = _TeiParser()

    def map(
        self,
        tei_xml: str,
        docling_result: DoclingResult,
        *,
        page_height_pt: float | None = None,
    ) -> FusedDocument:
        """Fuse GROBID TEI XML with Docling assets.

        Args:
            tei_xml: Raw TEI XML string from ``GrobidResult.tei_xml``.
            docling_result: Result from ``DoclingExecutor.process_pdf()``.
            page_height_pt: Actual PDF page height in points. Overrides the
                constructor default for this call only.

        Returns:
            ``FusedDocument`` — always returned even if matching yields zero hits.
        """
        page_height = page_height_pt if page_height_pt is not None else self._page_height
        grobid_doc = self._parser.parse(tei_xml)
        fused_assets = self._match_assets(
            docling_result.assets,
            grobid_doc.figure_refs,
            page_height=page_height,
        )
        return FusedDocument(grobid=grobid_doc, assets=fused_assets)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_assets(
        self,
        assets: list[ExtractedAsset],
        fig_refs: list[GrobidFigureRef],
        *,
        page_height: float,
    ) -> list[FusedAsset]:
        """Global greedy assignment of assets to GROBID figure refs."""
        if not fig_refs:
            logger.warning("No GROBID figure refs found; all assets will be unmatched.")
            return [FusedAsset(asset=a) for a in assets]

        # Score matrix: scores[ai][ri]
        scores: list[list[float]] = [
            [self._score(asset, ref, page_height=page_height) for ref in fig_refs]
            for asset in assets
        ]

        used_refs: set[int] = set()
        assignment: dict[int, tuple[int, float]] = {}  # asset_idx → (ref_idx, score)

        # Greedy: repeatedly consume the globally highest unassigned pair
        while True:
            best: float = 0.0
            best_ai = best_ri = -1
            for ai, row in enumerate(scores):
                if ai in assignment:
                    continue
                for ri, s in enumerate(row):
                    if ri in used_refs:
                        continue
                    if s > best:
                        best = s
                        best_ai = ai
                        best_ri = ri
            if best <= 0.0:
                break
            assignment[best_ai] = (best_ri, best)
            used_refs.add(best_ri)

        fused: list[FusedAsset] = []
        for ai, asset in enumerate(assets):
            if ai in assignment:
                ri, conf = assignment[ai]
                fused.append(FusedAsset(
                    asset=asset,
                    grobid_ref=fig_refs[ri],
                    match_confidence=conf,
                ))
            else:
                fused.append(FusedAsset(asset=asset))

        matched = sum(1 for f in fused if f.grobid_ref is not None)
        logger.info(
            "Matched %d / %d assets to %d GROBID refs (%d refs unused).",
            matched, len(assets), len(fig_refs), len(fig_refs) - len(used_refs),
        )
        # Debug unmatched so callers can diagnose
        for fa in fused:
            if fa.grobid_ref is None:
                logger.debug(
                    "Unmatched asset: type=%s page=%s bbox=%s caption=%r",
                    fa.asset.asset_type,
                    fa.asset.page_number,
                    fa.asset.bbox,
                    fa.asset.caption,
                )
        return fused

    def _flip_y(
        self,
        bbox: tuple[float, float, float, float],
        page_height: float,
    ) -> tuple[float, float, float, float]:
        """Flip Docling bottom-left bbox to top-left origin.

        Input  (Docling): x1, y1_bottom, x2, y2_bottom
        Output (GROBID):  x1, y1_top,   x2, y2_top

        y_top = page_height - y_bottom  (for each y coordinate)
        y1_top is the smaller value after flipping, so we swap:
            new_y1 = page_height - y2_bottom
            new_y2 = page_height - y1_bottom
        """
        x1, y1, x2, y2 = bbox
        return (x1, page_height - y2, x2, page_height - y1)

    def _score(
        self,
        asset: ExtractedAsset,
        ref: GrobidFigureRef,
        *,
        page_height: float,
    ) -> float:
        """Compute match confidence in [0, 1] for one (asset, ref) pair."""
        score = 0.0

        same_page = (
            asset.page_number is not None
            and ref.coords is not None
            and asset.page_number == ref.coords[0]
        )

        # ------------------------------------------------------------------
        # 1 & 2. Spatial: IoU after Y-axis normalisation
        # ------------------------------------------------------------------
        if same_page and ref.coords is not None:
            _, rx1, ry1, rx2, ry2 = ref.coords

            if asset.bbox is not None:
                # Flip Docling bbox so both share top-left origin
                ax1, ay1, ax2, ay2 = self._flip_y(asset.bbox, page_height)
                iou = _iou((ax1, ay1, ax2, ay2), (rx1, ry1, rx2, ry2))
                if iou >= self._iou_threshold:
                    # Strong spatial hit — short-circuit, no need to check further
                    return min(1.0, 0.60 + iou * 0.40)
                if iou > 0.0:
                    score = max(score, 0.25 + iou * 0.35)
            else:
                # Same page but Docling has no bbox (e.g. formula)
                score = max(score, 0.15)

        # ------------------------------------------------------------------
        # 3. Same page, GROBID has no coords
        # ------------------------------------------------------------------
        if (
            asset.page_number is not None
            and ref.coords is None
            and asset.page_number is not None
        ):
            # Can't do spatial — give a very small prior for same-page
            score = max(score, 0.10)

        # ------------------------------------------------------------------
        # 4. Label ordinal match  ("Figure 2" label ↔ "2" in caption)
        # ------------------------------------------------------------------
        ref_nums = _ordinal_numbers(ref.label or "")
        asset_nums = _ordinal_numbers(asset.caption or "")
        if ref_nums and asset_nums and ref_nums[0] == asset_nums[0]:
            bonus = 0.70 if same_page else 0.55
            score = max(score, bonus)

        # ------------------------------------------------------------------
        # 5. Caption token similarity
        # ------------------------------------------------------------------
        cap_sim = _caption_similarity(asset.caption, ref.caption)
        if cap_sim >= self._caption_threshold:
            score = max(score, 0.35 + cap_sim * 0.55)

        return score