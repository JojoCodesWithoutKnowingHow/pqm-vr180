import pytest

#: The CLI's defaults before 0.5.0. The outpaint rounds' tests (test_outpaint.py,
#: test_round2.py) were written against them, each passing only its round's flags;
#: they keep testing what they tested by running on these.
V1_DEFAULTS = dict(
    method="noob", denoise=None, join="blend", layout="none", layout_hires=0,
    layout_strong=False, extend_side=0.0, touch_mask_weight_j=None, layout_mask_grow=2,
    layout_mask_blur=None, auto_pipeline=False, adetail=0.0, compose=-1.0, seam_repaint=0.0,
    soften_rim=12, detail_match=True,
)


@pytest.fixture
def v1_defaults(monkeypatch):
    from vr180 import cli

    def parse(argv=None):
        p = cli.parser()
        p.set_defaults(**V1_DEFAULTS)
        return p.parse_args(argv)

    monkeypatch.setattr(cli, "parse", parse)
