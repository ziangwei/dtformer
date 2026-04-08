"""DTFormer segmentor (top-level model).

Assembles encoder (DTFormerEncoder + TSA-E), decoder (HSG + TSA-D),
and segmentation loss into a single nn.Module.
"""
