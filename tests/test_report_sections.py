"""Keyword snapshots of every report section and the section numbering."""

import json
from pathlib import Path

import pytest

from msp.report import SECTIONS, _fmt_cell, _number_sections, generate_report

PNG = b"\x89PNG\r\n\x1a\nfake"


def full_run(root: Path) -> None:
    (root / "figures").mkdir()
    for name in (
        "umap_batch.png",
        "umap_msp_leiden_r1.0.png",
        "umap__ann_coarse.png",
        "standissect_product.png",
        "qc_umap_pct_counts_mt.png",
        "qc_violin_pct_counts_mt.png",
        "fractal_marker_heatmap.png",
        "leiden_qc_violin_doublet_score_msp_leiden_r1.0.png",
        "umap_preannotation_removal.png",
        "inspect_umap_action.png",
        "annotation_umap_coarse.png",
        "annotation_umap_fine.png",
        "annotation_umap_removed.png",
    ):
        (root / "figures" / name).write_bytes(PNG)
    (root / "integration_summary.csv").write_text("n_cells,20\nn_samples,2\n")
    (root / "per_sample_qc.csv").write_text(
        "batch,n_cells,median_pct_counts_mt,median_n_genes_by_counts\nA,10,3.14159,1234.56\nB,10,2.71828,987.65\n"
    )
    (root / "sample_decisions.csv").write_text(
        "sample,decision,n_cells,reason\nA,include,10,ok\nB,exclude,10,too few\n"
    )
    (root / "cluster_qc_standissect_product.csv").write_text(
        "standissect_product,n_cells,n_samples\nc0_0,15,2\nc0_1,5,1\n"
    )
    (root / "minor_sibling_qc.csv").write_text(
        "subcluster,parent,n_cells,core_n_cells,frac_of_core,status,pct_drop_upstream,doublet_median,"
        "doublet_significant,n_hits,recommend_removal\n"
        "c0_1,0,5,15,0.333,tested,0.0,0.4,True,1.0,True\n"
        "c0_2,0,6,15,0.4,big_sibling_skip,,,,,\n"
    )
    (root / "fractal_markers.csv").write_text("parent,gene,rank,logfoldchange,pvals_adj\n0,MARKERGENE,1,2.5,1e-5\n")
    (root / "cluster_qc_msp_leiden_r1.0.csv").write_text("msp_leiden_r1.0,n_cells,n_samples\n0,20,2\n")
    (root / "cell_outlier_summary.csv").write_text(
        "key,cluster,n_cells,n_doublet_score_outlier,pct_doublet_score_outlier,n_recommend_removal,pct_recommend_removal\n"
        "msp_leiden_r1.0,0,20,1,5.0,1,5.0\n"
    )
    (root / "deg_global_msp_leiden_r1.0.csv").write_text(
        "group,names,logfoldchanges,pvals_adj,pct1,pct2\n0,FOS,2.0,1e-5,0.9,0.1\n1,ACTB,1.5,1e-4,0.8,0.2\n"
    )
    (root / "deg_local_msp_leiden_r1.0.csv").write_text(
        "group,names,logfoldchanges,pvals_adj,pct1,pct2,neighbors\n0,FOS,1.0,1e-3,0.9,0.5,1\n"
    )
    (root / "stress_clusters.csv").write_text(
        "key,cluster,view,n_hits,hit_genes,stress,recommend_removal\n"
        "msp_leiden_r1.0,0,global,4,FOS,True,True\nmsp_leiden_r1.0,0,local,1,FOS,False,True\n"
        "msp_leiden_r1.0,1,global,0,,False,False\n"
    )
    (root / "inspection_proposal.json").write_text(
        json.dumps(
            {
                "cluster_key": "msp_leiden_r1.0",
                "clusters": [
                    {
                        "cluster": "0",
                        "verdict": "artifact-lowquality",
                        "action": "drop",
                        "confidence": "high",
                        "tests": {
                            "markers": "stress only",
                            "qc": "high mt",
                            "composition": "all",
                            "geometry": "edge",
                            "stability": "stable",
                        },
                        "rationale": "INSPECT_RATIONALE",
                    }
                ],
                "overall": "INSPECT_OVERALL",
            }
        )
    )
    (root / "annotation_proposal.json").write_text(
        json.dumps(
            {
                "cluster_key": "msp_leiden_r2.0",
                "merged_groups": ["1+2"],
                "overall": "ANNOTATE_OVERALL",
                "clusters": [
                    {
                        "cluster_id": "1",
                        "coarse_label": "Immune",
                        "fine_label": "T cell",
                        "merge_target": "2",
                        "action": "keep",
                        "remove_reason": None,
                        "confidence": "medium",
                        "evidence": {"distinctness": "EV_DISTINCT", "markers": "CD3D", "merge": "same"},
                        "rationale": "ANNOTATE_RATIONALE",
                    }
                ],
            }
        )
    )
    (root / "annotation_removed.csv").write_text(
        "cell,msp_leiden_r2.0,preannotation,inspect_drop,annotate_remove,remove_reason\n"
        "c1,0,True,True,False,\nc2,0,False,True,False,\n"
    )


@pytest.fixture
def report_text(tmp_path):
    full_run(tmp_path)
    return Path(generate_report(tmp_path)).read_text()


def test_every_section_renders_in_order_with_consecutive_numbers(report_text):
    positions = []
    for number, section in enumerate(SECTIONS, start=1):
        heading = f'<h2 id="{section.anchor}">{number}. {section.label}</h2>'
        assert heading in report_text, heading
        assert f'href="#{section.anchor}">{number}. ' in report_text
        positions.append(report_text.index(heading))
    assert positions == sorted(positions)
    assert report_text.count("<h2 ") == len(SECTIONS)


@pytest.mark.parametrize(
    "keyword",
    [
        "Sample inclusion (1/2 entered integration)",  # sample summary
        "Per-sample QC",
        "Inherited annotations from One-sample Pipeline (OSP)",  # umaps
        "Leiden clusterings",
        "standissect clusters",
        "QC metrics",
        "StanDissect Minor sibling fractals QC",  # per-cluster QC
        "1/2 siblings recommend_removal, 1 stats-tested",
        "StanDissect Per-cluster QC violins",
        "Fractal marker dot plot",
        "MARKERGENE",
        "Cell-level doublet / ambient-RNA outliers",  # leiden QC
        "Per-cluster cutoff violins",
        "Pre-annotation filtering",  # cluster annotations
        "msp_leiden_r1.0 — global",
        "msp_leiden_r1.0 — local",
        "[recommend_removal]",
        "Per-cluster verdicts and actions",  # inspection
        "Five-test evidence",
        "INSPECT_RATIONALE",
        "INSPECT_OVERALL",
        "Annotated UMAPs (removed cells excluded)",  # annotation
        "Removed cells (all sources)",
        "2 cells removed — by source (a cell may have several): preannotation=1, inspect_drop=2, annotate_remove=0",
        "Per-cluster decisions",
        "Merged groups:</b> 1+2",
        "EV_DISTINCT",
        "ANNOTATE_OVERALL",
    ],
)
def test_section_keywords(report_text, keyword):
    assert keyword in report_text


def test_stress_hit_genes_are_highlighted_and_display_rounding_is_applied(report_text):
    assert '<b style="color:#c0392b">FOS</b> (2.0)' in report_text  # global view hit gene
    assert "ACTB (1.5)" in report_text and 'color:#c0392b">ACTB' not in report_text
    assert report_text.count("[recommend_removal]</span>") == 2  # merged verdict shows on both views of cluster 0
    assert (
        "<td>3.14</td>" in report_text and "<td>1235</td>" in report_text
    )  # medians / gene counts rounded for display
    assert "3.14159" not in report_text
    assert 'style="color:#8b0000;font-weight:bold"' in report_text  # recommend_removal sibling row
    assert 'style="color:#888"' in report_text  # big sibling row


def test_numbering_closes_gaps_and_skips_empty_sections(tmp_path):
    (tmp_path / "integration_summary.csv").write_text("n_cells,20\n")
    (tmp_path / "deg_global_k.csv").write_text("group,names,logfoldchanges,pvals_adj,pct1,pct2\n0,G,1.0,0.1,0.5,0.1\n")
    text = Path(generate_report(tmp_path)).read_text()
    assert '<h2 id="sample-summary">1. Sample Summary</h2>' in text
    assert '<h2 id="deg">2. Cluster Annotations</h2>' in text
    assert text.count("<h2 ") == 2 and "3. " not in text
    assert "Per-cluster QC" not in text and "UMAPs" not in text


def test_empty_directory_renders_no_sections_and_no_toc(tmp_path):
    text = Path(generate_report(tmp_path)).read_text()
    assert "<h2 " not in text and "<nav" not in text


def test_number_sections_drops_empty_bodies_and_anchors_subsections():
    htmls, toc = _number_sections(
        [("sample-summary", ""), ("deg", "<h3>Tab A</h3><p>x</p>"), ("annotation", "<p>y</p>")]
    )
    assert htmls == [
        '<h2 id="deg">1. Cluster Annotations</h2><h3 id="deg-1">Tab A</h3><p>x</p>',
        '<h2 id="annotation">2. Cell Type Annotation</h2><p>y</p>',
    ]
    assert 'href="#deg">1. Cluster Annotations' in toc and 'href="#deg-1">Tab A' in toc
    assert 'href="#annotation">2. Cell Type Annotation' in toc
    assert _number_sections([]) == ([], "")


def test_fmt_cell_rounds_only_medians_and_gene_counts():
    assert _fmt_cell("median_pct_counts_mt", "3.14159") == "3.14"
    assert _fmt_cell("median_n_genes_by_counts", "1234.56") == "1235"
    assert _fmt_cell("n_cells", "1234.56") == "1234.56"
    assert _fmt_cell("sample", "<A&B>") == "&lt;A&amp;B&gt;"
